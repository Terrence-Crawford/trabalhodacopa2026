import pandas as pd
import numpy as np

def calculate_form(matches_df, team_name, n_games=10):
    """Calcula o aproveitamento percentual de pontos nos últimos N jogos."""
    # Filtra partidas onde o time jogou (padrão de colunas comum em APIs)
    team_matches = matches_df[(matches_df['home_team'] == team_name) | (matches_df['away_team'] == team_name)]
    
    if 'date' in team_matches.columns:
        team_matches = team_matches.sort_values(by='date', ascending=False)
        
    recent_games = team_matches.head(n_games)
    
    if recent_games.empty:
        return 0.5
        
    points = 0
    for _, match in recent_games.iterrows():
        if match['home_team'] == team_name:
            if match['home_score'] > match['away_score']: 
                points += 1
            elif match['home_score'] == match['away_score']: 
                points += 0.5
        else:
            if match['away_score'] > match['home_score']: 
                points += 1
            elif match['home_score'] == match['away_score']: 
                points += 0.5
                
    return points / len(recent_games)


def calculate_defense_and_attack(matches_df, team_name, n_games=10):
    """Retorna a média de gols pró (ataque) e gols sofridos (defesa) do histórico recente."""
    team_matches = matches_df[(matches_df['home_team'] == team_name) | (matches_df['away_team'] == team_name)]
    
    if 'date' in team_matches.columns:
        team_matches = team_matches.sort_values(by='date', ascending=False)
        
    recent_games = team_matches.head(n_games)
    
    if recent_games.empty:
        return 0.0, 0.0
        
    scored = 0
    conceded = 0
    
    for _, match in recent_games.iterrows():
        if match['home_team'] == team_name:
            scored += match['home_score']
            conceded += match['away_score']
        else:
            scored += match['away_score']
            conceded += match['home_score']
            
    return (scored / len(recent_games)), (conceded / len(recent_games))


def extract_squad_advanced_metrics(df_api_stats, team_name):
    """
    Extrai os indicadores consolidados da seleção vindos das tabelas da API:
    Elo, Ranking FIFA, Estilo de Jogo (Posse/Agressividade) e Valor de Mercado Médio.
    """
    try:
        team_data = df_api_stats.loc[df_api_stats['team'] == team_name].iloc[0]
        elo_val = team_data.get('elo', 1500)
        fifa_rank = team_data.get('ranking_fifa', 50)
        estilo_jogo = team_data.get('estilo_jogo', 5.0) # Ex: Média de posse de bola ou finalizações
        valor_medio = team_data.get('valor_medio_jogador', 10.0) # Valor em milhões
    except (IndexError, KeyError):
        # Valores padrão neutros para seleções ausentes na amostragem
        elo_val = 1500
        fifa_rank = 50
        estilo_jogo = 5.0
        valor_medio = 10.0
        
    return elo_val, fifa_rank, estilo_jogo, valor_medio


def gerar_vetor_confronto(team_a, team_b, matches_df, df_api_stats):
    """Gera o vetor de diferenças técnicas brutas (A - B) para alimentar o modelo e gráficos."""
    # 1. Métricas de Momento (Últimos jogos)
    form_a = calculate_form(matches_df, team_a)
    form_b = calculate_form(matches_df, team_b)
    
    # 2. Ataque e Defesa (Médias de gols)
    ataque_a, defesa_a = calculate_defense_and_attack(matches_df, team_a)
    ataque_b, defesa_b = calculate_defense_and_attack(matches_df, team_b)
    
    # 3. Métricas Estruturais da API (Elo, FIFA, Estilo, Valor)
    elo_a, fifa_a, estilo_a, valor_a = extract_squad_advanced_metrics(df_api_stats, team_a)
    elo_b, fifa_b, estilo_b, valor_b = extract_squad_advanced_metrics(df_api_stats, team_b)
    
    # Retorna o dicionário de features diferenciais relacionando as variáveis solicitadas
    return {
        'dif_aproveitamento': form_a - form_b,
        'dif_gols_marcados': ataque_a - ataque_b,
        'dif_defesa_gols_sofridos': defesa_a - defesa_b, # Valores negativos indicam melhor defesa para A
        'dif_elo': elo_a - elo_b,
        'dif_ranking_fifa': fifa_b - fifa_a, # Invertido pois menor ranking na FIFA significa time melhor
        'dif_estilo_jogo': estilo_a - estilo_b,
        'dif_valor_medio_elenco': valor_a - valor_b
    }

def simular_confronto_direto(time, adversario, df):
    """
    Compara dois times usando as variáveis disponíveis
    e conta quantas vantagens cada um possui.
    """

    dados_time = df.loc[time]
    dados_adv = df.loc[adversario]

    vantagens_time = 0
    vantagens_adv = 0


    # Elo maior indica maior força histórica
    if dados_time['elo'] > dados_adv['elo']:
        vantagens_time += 1
    else:
        vantagens_adv += 1


    # Ranking FIFA menor é melhor
    if dados_time['ranking_fifa'] < dados_adv['ranking_fifa']:
        vantagens_time += 1
    else:
        vantagens_adv += 1


    # Estilo de jogo
    if dados_time['estilo_jogo'] > dados_adv['estilo_jogo']:
        vantagens_time += 1
    else:
        vantagens_adv += 1


    # Defesa: menor número significa melhor defesa
    if dados_time['defesa'] < dados_adv['defesa']:
        vantagens_time += 1
    else:
        vantagens_adv += 1


    # Valor do elenco
    if dados_time['valor_elenco_milhoes'] > dados_adv['valor_elenco_milhoes']:
        vantagens_time += 1
    else:
        vantagens_adv += 1


    # Forma recente
    if dados_time['aproveitamento_recente'] > dados_adv['aproveitamento_recente']:
        vantagens_time += 1
    else:
        vantagens_adv += 1


    return vantagens_time, vantagens_adv