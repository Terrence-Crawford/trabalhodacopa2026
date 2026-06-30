import numpy as np
import pandas as pd

def simular_placar_jogo(stats_a, stats_b):
    """
    Calcula o placar com base nas forças relativas obtidas via Elo e Overall.
    Retorna (gols_a, gols_b).
    """
    # Força ofensiva vs defensiva combinada
    lambda_a = np.exp((stats_a['elo'] - stats_b['elo']) / 800 + (stats_a['overall_titulares'] - 75) * 0.05)
    lambda_b = np.exp((stats_b['elo'] - stats_a['elo']) / 800 + (stats_b['overall_titulares'] - 75) * 0.05)
    
    # Limitação de taxas excessivas para manter a média realista de gols
    lambda_a = max(0.2, min(4.0, lambda_a))
    lambda_b = max(0.2, min(4.0, lambda_b))
    
    gols_a = np.random.poisson(lambda_a)
    gols_b = np.random.poisson(lambda_b)
    
    return gols_a, gols_b

def resolver_mata_mata(stats_a, stats_b, nome_a, nome_b):
    """
    Executa a simulação de um jogo eliminatório. Em caso de empate,
    simula prorrogação e, persistindo, decisão por pênaltis.
    """
    g_a, g_b = simular_placar_jogo(stats_a, stats_b)
    
    if g_a != g_b:
        return nome_a if g_a > g_b else nome_b
        
    # Prorrogação (tempo extra)
    p_a = np.random.poisson(stats_a['overall_titulares'] * 0.005)
    p_b = np.random.poisson(stats_b['overall_titulares'] * 0.005)
    
    if p_a != p_b:
        return nome_a if p_a > p_b else nome_b
        
    # Decisão por Pênaltis (baseado na experiência/overall e fator estocástico)
    prob_disputa_a = stats_a['overall_titulares'] / (stats_a['overall_titulares'] + stats_b['overall_titulares'])
    vencedor = np.random.choice([nome_a, nome_b], p=[prob_disputa_a, 1 - prob_disputa_a])
    return vencedor

def executar_monte_carlo_2026(df_selecoes, estrutura_grupos, n_simulacoes=10000):
    """
    Simula o fluxo completo da Copa do Mundo 2026 com 48 seleções organizadas
    em 12 grupos de 4 equipes. Avançam os 2 melhores de cada grupo + 8 melhores terceiros.
    """
    fases = ['Dezesseisavos', 'Oitavas', 'Quartas', 'Semi', 'Final', 'Campeão']
    tracker = {time: {fase: 0 for fase in fases} for time in df_selecoes.index}
    
    for _ in range(n_simulacoes):
        classificados_grupo = []
        terceiros_colocados = {}
        
        # 1. Fase de Grupos
        for grupo, times in estrutura_grupos.items():
            pontos = {t: 0 for t in times}
            saldos = {t: 0 for t in times}
            
            # Confrontos internos do grupo
            for i in range(len(times)):
                for j in range(i + 1, len(times)):
                    t1, t2 = times[i], times[j]
                    g1, g2 = simular_placar_jogo(df_selecoes.loc[t1], df_selecoes.loc[t2])
                    
                    if g1 > g2:
                        pontos[t1] += 3
                    elif g2 > g1:
                        pontos[t2] += 3
                    else:
                        pontos[t1] += 1
                        pontos[t2] += 1
                    saldos[t1] += (g1 - g2)
                    saldos[t2] += (g2 - g1)
                    
            # Ranqueamento do grupo utilizando Elo como critério de desempate secundário
            ranking_grupo = sorted(times, key=lambda t: (pontos[t], saldos[t], df_selecoes.loc[t, 'elo']), reverse=True)
            
            classificados_grupo.extend(ranking_grupo[:2])
            terceiros_colocados[ranking_grupo[2]] = (pontos[ranking_grupo[2]], saldos[ranking_grupo[2]], df_selecoes.loc[ranking_grupo[2], 'elo'])
            
        # Seleção dos 8 melhores terceiros colocados gerais
        melhores_terceiros = sorted(terceiros_colocados.keys(), key=lambda t: terceiros_colocados[t], reverse=True)[:8]
        top_32 = classificados_grupo + melhores_terceiros
        
        for t in top_32: 
            tracker[t]['Dezesseisavos'] += 1
            
        # 2. Fase Eliminatória (Mata-Mata Corrente)
        # Rodada de 32 (Dezesseisavos) -> Oitavas
        oitavas = []
        for i in range(0, 32, 2):
            t1, t2 = top_32[i], top_32[i+1]
            vencedor = resolver_mata_mata(df_selecoes.loc[t1], df_selecoes.loc[t2], t1, t2)
            oitavas.append(vencedor)
            tracker[vencedor]['Oitavas'] += 1
            
        # Oitavas -> Quartas
        quartas = []
        for i in range(0, 16, 2):
            t1, t2 = oitavas[i], oitavas[i+1]
            vencedor = resolver_mata_mata(df_selecoes.loc[t1], df_selecoes.loc[t2], t1, t2)
            quartas.append(vencedor)
            tracker[vencedor]['Quartas'] += 1
            
        # Quartas -> Semi
        semi = []
        for i in range(0, 8, 2):
            t1, t2 = quartas[i], quartas[i+1]
            vencedor = resolver_mata_mata(df_selecoes.loc[t1], df_selecoes.loc[t2], t1, t2)
            semi.append(vencedor)
            tracker[vencedor]['Semi'] += 1
            
        # Semi -> Final
        final = []
        for i in range(0, 4, 2):
            t1, t2 = semi[i], semi[i+1]
            vencedor = resolver_mata_mata(df_selecoes.loc[t1], df_selecoes.loc[t2], t1, t2)
            final.append(vencedor)
            tracker[vencedor]['Final'] += 1
            
        # Final -> Campeão
        campeao = resolver_mata_mata(df_selecoes.loc[final[0]], df_selecoes.loc[final[1]], final[0], final[1])
        tracker[campeao]['Campeão'] += 1
        
    df_resultados = pd.DataFrame(tracker).T / n_simulacoes * 100
    return df_resultados