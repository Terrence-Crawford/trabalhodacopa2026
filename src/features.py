import pandas as pd
import numpy as np

def calcular_força_historica(matches_df, n_anos=10):
    """
    Calcula o Elo histórico removendo jogos futuros/NAs do all_matches.csv
    """
    df_recente = matches_df.copy()
    df_recente['date'] = pd.to_datetime(df_recente['date'])
    
    # 1. Filtro de tempo (últimos 10 anos)
    limite_data = df_recente['date'].max() - pd.DateOffset(years=n_anos)
    df_recente = df_recente[df_recente['date'] >= limite_data].copy()
    
    # 2. LIMPEZA CRUCIAL: Remove jogos futuros (onde o placar é NA)
    df_recente = df_recente.dropna(subset=['home_score', 'away_score'])
    
    # Converte placares para inteiros para evitar problemas de tipo
    df_recente['home_score'] = df_recente['home_score'].astype(int)
    df_recente['away_score'] = df_recente['away_score'].astype(int)
    
    times = set(df_recente['home_team'].unique()).union(set(df_recente['away_team'].unique()))
    elo_dict = {time: 1500.0 for time in times}
    
    K = 32
    for _, row in df_recente.sort_values('date').iterrows():
        t1, t2 = row['home_team'], row['away_team']
        g1, g2 = row['home_score'], row['away_score']
        
        elo1, elo2 = elo_dict[t1], elo_dict[t2]
        exp1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
        exp2 = 1 - exp1
        
        if g1 > g2:
            res1, res2 = 1.0, 0.0
        elif g1 < g2:
            res1, res2 = 0.0, 1.0
        else:
            res1, res2 = 0.5, 0.5
            
        elo_dict[t1] = elo1 + K * (res1 - exp1)
        elo_dict[t2] = elo2 + K * (res2 - exp2)
        
    return elo_dict

def extrair_metricas_elenco(players_df, team_name):
    """
    Busca o OVR dos jogadores no male_players.csv
    """
    squad = players_df[players_df['Nation'] == team_name]
    if squad.empty:
        return 70.0, 0
        
    top_11 = squad.sort_values(by='OVR', ascending=False).head(11)
    return float(top_11['OVR'].mean()), int(len(squad[squad['OVR'] >= 85]))

def obter_pontos_fifa(ranking_df, team_name):
    """
    Busca a última pontuação de pontos FIFA do fifa_ranking_2024.csv
    """
    pais_df = ranking_df[ranking_df['country_full'] == team_name]
    if pais_df.empty:
        return 1300.0
        
    pais_df = pais_df.copy()
    pais_df['rank_date'] = pd.to_datetime(pais_df['rank_date'])
    linha_recente = pais_df.sort_values(by='rank_date', ascending=False).iloc[0]
    return float(linha_recente['total_points'])