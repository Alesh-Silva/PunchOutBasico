import pygame, sys, os, joblib, numpy as np, pandas as pd
from collections import deque
from sklearn.tree import DecisionTreeClassifier

# Configurações e cores
pygame.init()
LARGURA, ALTURA = 700, 500
tela = pygame.display.set_mode((LARGURA, ALTURA))
pygame.display.set_caption("Punch Out 2D - IA Adaptativa")
relogio = pygame.time.Clock()
fonte = pygame.font.SysFont("arial", 16)
fonte_grande = pygame.font.SysFont("arial", 30, bold=True)

BRANCO, PRETO, AZUL = (255, 255, 255), (20, 20, 20), (50, 120, 220)
VERMELHO, VERDE, AMARELO, CINZA = (220, 50, 50), (40, 200, 70), (240, 200, 40), (70, 70, 70)

# Modelo de IA
modelo_ia = None
if os.path.exists("modelo_punchout.joblib"):
    try: modelo_ia = joblib.load("modelo_punchout.joblib")["modelo"]
    except Exception: pass
if modelo_ia is None:
    modelo_ia = DecisionTreeClassifier(max_depth=4).fit(np.random.randint(0, 6, (200, 5)), np.random.randint(0, 6, 200))

# Variáveis de jogo e progressão
round_atual, nivel_ia = 1, 1.0
historico = deque([0, 1, 0], maxlen=3)
p_hp, p_estado, p_timer, cooldown, stamina = 100, "neutro", 0, 0, 5
e_hp, e_estado, e_timer, e_ataque_timer = 100, "neutro", 0, 80
bloqueios_seguidos, msg, msg_timer, vencedor = 0, "Round 1! Lute!", 40, ""

def reiniciar_round():
    global p_hp, p_estado, p_timer, cooldown, stamina, e_hp, e_estado, e_timer, e_ataque_timer
    global bloqueios_seguidos, round_atual, nivel_ia, msg, msg_timer, vencedor, historico
    p_hp, p_estado, p_timer, cooldown, stamina = 100, "neutro", 0, 0, 5
    e_hp, e_estado, e_timer = 100, "neutro", 0
    bloqueios_seguidos = 0
    
    if "NOCAUTEADO" in vencedor:
        round_atual, nivel_ia = 1, 1.0
        historico = deque([0, 1, 0], maxlen=3)
        msg, msg_timer = "Derrota! Voltou ao Round 1 (IA Resetada)", 45
    else:
        round_atual += 1
        nivel_ia = min(4.5, 1.0 + (round_atual - 1) * 0.5)
        msg, msg_timer = f"Round {round_atual}! IA Nível {nivel_ia:.1f}x", 45

    vencedor = ""
    e_ataque_timer = max(40, int(80 - (nivel_ia * 8)))

# Loop principal
while True:
    relogio.tick(60)

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT: pygame.quit(); sys.exit()
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_r: reiniciar_round()
            if vencedor == "" and p_estado == "neutro" and cooldown == 0:
                if ev.key in [pygame.K_a, pygame.K_d]:
                    acao = 0 if ev.key == pygame.K_a else 1
                    p_estado, p_timer, cooldown = ("soco_esq" if acao == 0 else "soco_dir"), 14, 18
                    
                    entrada = pd.DataFrame([{"hp_jogador": p_hp, "stamina_jogador": 100, "acao_t_3": historico[0], "acao_t_2": historico[1], "acao_t_1": historico[2]}])
                    pred = int(modelo_ia.predict(entrada)[0])
                    historico.append(acao)

                    if e_estado == "aberto":
                        # IA desvia de contra-ataques a partir do nivel 3.0x
                        chance_esquiva_contra = min(0.90, 0.35 + (nivel_ia - 3.0) * 0.55) if nivel_ia >= 3.0 and pred == acao else 0
                        if np.random.rand() < chance_esquiva_contra:
                            e_estado, e_timer, msg, msg_timer = "esquiva", 18, "IA Desviou do Contra-ataque!", 25
                        else:
                            e_hp -= 15
                            msg, msg_timer, bloqueios_seguidos = "Contra-ataque! -15", 25, 0
                    elif e_estado in ["neutro", "defesa"]:
                        if pred == acao and np.random.rand() < min(0.75, 0.25 * nivel_ia):
                            e_estado, e_timer, msg, msg_timer = "esquiva", 18, "IA Esquivou!", 25
                        else:
                            stamina = max(0, stamina - 1)
                            msg, msg_timer = "Bloqueado!", 20
                            if stamina == 0:
                                e_estado, e_timer, p_hp, stamina = "contra_ataque", 16, p_hp - 12, 5
                                msg, msg_timer = "Sem Stamina! Atingido! -12", 25
                elif ev.key in [pygame.K_SPACE, pygame.K_w]:
                    p_estado, p_timer, cooldown = "esquiva", 22, 12
                    historico.append(3)
                elif ev.key == pygame.K_s:
                    p_estado, p_timer, cooldown = "defesa", 24, 10
                    historico.append(5)

    # Atualizações de timers
    if cooldown > 0: cooldown -= 1
    if p_timer > 0:
        p_timer -= 1
        if p_timer == 0: p_estado = "neutro"
    if e_timer > 0:
        e_timer -= 1
        if e_timer == 0:
            if e_estado == "esquiva":
                e_estado, e_timer, p_hp, stamina = "contra_ataque", 14, p_hp - 14, 5
                msg, msg_timer = "IA Esquivou e Bateu!", 25
            else: e_estado = "neutro"

    # Lógica de ataque do inimigo
    if vencedor == "" and e_estado in ["neutro", "aviso"]:
        e_ataque_timer -= 1
        if e_ataque_timer <= max(18, int(26 / (nivel_ia ** 0.5))) and e_ataque_timer > 0:
            e_estado = "aviso"
        elif e_ataque_timer <= 0:
            e_estado, e_timer = "atacando", 15
            e_ataque_timer = max(45, int(85 - (nivel_ia * 15)))
            if p_estado == "esquiva":
                e_estado, e_timer, msg, msg_timer = "aberto", max(24, int(40 - (nivel_ia * 6))), "Errou! Bata agora!", 30
            elif p_estado == "defesa":
                p_hp -= 4
                msg, msg_timer, e_estado = "Bloqueou!", 20, "neutro"
            else:
                p_hp, stamina = p_hp - 16, 5
                msg, msg_timer, e_estado = "Atingido! -16", 25, "neutro"

    if p_hp <= 0 and vencedor == "": p_hp, vencedor = 0, "VOCÊ FOI NOCAUTEADO!"
    elif e_hp <= 0 and vencedor == "": e_hp, vencedor = 0, "VITÓRIA! K.O.!"
    if msg_timer > 0: msg_timer -= 1

    # Renderização gráfica
    tela.fill(PRETO)
    pygame.draw.rect(tela, (40, 45, 60), (50, 110, LARGURA - 100, ALTURA - 130))
    for i, c in enumerate([VERMELHO, BRANCO, AZUL]): pygame.draw.line(tela, c, (40, 150 + i * 40), (LARGURA - 40, 150 + i * 40), 3)

    # Barras de vida e HUD
    pygame.draw.rect(tela, CINZA, (50, 25, 200, 18))
    pygame.draw.rect(tela, VERDE, (50, 25, int((p_hp / 100) * 200), 18))
    tela.blit(fonte.render(f"Você: {p_hp}  |  Stamina: {stamina}", True, BRANCO), (50, 5))

    pygame.draw.rect(tela, CINZA, (LARGURA - 250, 25, 200, 18))
    pygame.draw.rect(tela, VERMELHO, (LARGURA - 250, 25, int((e_hp / 100) * 200), 18))
    tela.blit(fonte.render(f"Inimigo (IA): {e_hp}", True, BRANCO), (LARGURA - 250, 5))

    txt_round = fonte.render(f"Round {round_atual} (Nível IA: {nivel_ia:.1f}x)", True, AMARELO)
    tela.blit(txt_round, (LARGURA // 2 - txt_round.get_width() // 2, 8))

    # Inimigo
    ex = LARGURA // 2 + (45 if e_estado == "esquiva" else (np.random.randint(-2, 3) if e_estado == "aberto" else 0))
    ey = 220
    cor_e = AMARELO if e_estado == "aviso" else ((240, 130, 60) if e_estado == "aberto" else VERMELHO)
    pygame.draw.rect(tela, cor_e, (ex - 30, ey - 30, 60, 60), border_radius=8)
    pygame.draw.circle(tela, (230, 190, 150), (ex, ey - 50), 20)
    
    # Luvas inimigo
    if e_estado in ["atacando", "contra_ataque"]:
        pygame.draw.circle(tela, (180, 20, 20), (ex - 35, ey + 25), 18); pygame.draw.circle(tela, (180, 20, 20), (ex + 35, ey + 25), 18)
    elif e_estado == "aberto":
        pygame.draw.circle(tela, (180, 20, 20), (ex - 42, ey + 15), 12); pygame.draw.circle(tela, (180, 20, 20), (ex + 42, ey + 15), 12)
    else:
        pygame.draw.circle(tela, (180, 20, 20), (ex - 18, ey - 35), 13); pygame.draw.circle(tela, (180, 20, 20), (ex + 18, ey - 35), 13)

    # Jogador
    px = LARGURA // 2 - (50 if p_estado == "esquiva" else 0)
    py = 380
    pygame.draw.rect(tela, AZUL, (px - 32, py - 30, 64, 60), border_radius=8)
    pygame.draw.circle(tela, (40, 40, 50), (px, py - 48), 18)
    
    # Luvas jogador
    if p_estado == "soco_esq":
        pygame.draw.circle(tela, VERDE, (px - 22, py - 85), 16); pygame.draw.circle(tela, VERDE, (px + 30, py - 20), 13)
    elif p_estado == "soco_dir":
        pygame.draw.circle(tela, VERDE, (px - 30, py - 20), 13); pygame.draw.circle(tela, VERDE, (px + 22, py - 85), 16)
    elif p_estado == "defesa":
        pygame.draw.circle(tela, VERDE, (px - 12, py - 60), 14); pygame.draw.circle(tela, VERDE, (px + 12, py - 60), 14)
    else:
        pygame.draw.circle(tela, VERDE, (px - 28, py - 25), 13); pygame.draw.circle(tela, VERDE, (px + 28, py - 25), 13)

    # Mensagens e telas de fim
    if msg_timer > 0 and vencedor == "":
        txt_m = fonte.render(msg, True, AMARELO if "Contra" in msg or "Errou" in msg else BRANCO)
        tela.blit(txt_m, (LARGURA // 2 - txt_m.get_width() // 2, 160))

    if vencedor != "":
        txt_v = fonte_grande.render(vencedor, True, AMARELO)
        tela.blit(txt_v, (LARGURA // 2 - txt_v.get_width() // 2, ALTURA // 2 - 30))
        txt_r_label = "Aperte [R] para Tentar de Novo (Volta ao Round 1)" if "NOCAUTEADO" in vencedor else "Aperte [R] para o Próximo Round"
        txt_r = fonte.render(txt_r_label, True, BRANCO)
        tela.blit(txt_r, (LARGURA // 2 - txt_r.get_width() // 2, ALTURA // 2 + 15))

    txt_c = fonte.render("[A] Soco Esq  |  [D] Soco Dir  |  [ESPAÇO] Esquivar  |  [S] Defender", True, (130, 130, 140))
    tela.blit(txt_c, (LARGURA // 2 - txt_c.get_width() // 2, ALTURA - 25))

    pygame.display.flip()
