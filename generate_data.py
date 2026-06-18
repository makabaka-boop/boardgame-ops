import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

players = [f'玩家{i:03d}' for i in range(1, 51)]
activities = ['狼人杀之夜', '三国杀争霸', '桌游嘉年华', '新手教学日', '周末欢乐局', 
              '策略桌游会', '卡牌对战赛', '家庭桌游日', '深夜桌游局', '节日特别场']
feedback_tags = ['规则复杂', '气氛热烈', '等待时间长', '场地舒适', '新手友好',
                 '时长合适', '奖品丰富', '组织混乱', '体验极佳', '建议改进']

start_date = datetime(2026, 1, 1)
date_list = [start_date + timedelta(days=i) for i in range(90)]

daily_activity_map = {}
for date in date_list:
    num_activities = random.randint(1, 2)
    day_activities = random.sample(activities, num_activities)
    daily_activity_map[date.strftime('%Y-%m-%d')] = day_activities

signups = []
for date_str, day_activities in daily_activity_map.items():
    for activity in day_activities:
        num_players = random.randint(4, min(20, len(players)))
        day_players = random.sample(players, num_players)
        for player in day_players:
            signups.append({
                '日期': date_str,
                '活动名': activity,
                '玩家': player,
                '桌号': random.randint(1, 5)
            })

signups_df = pd.DataFrame(signups)
signups_df.to_csv('data/signups.csv', index=False, encoding='utf-8-sig')

results = []
for date_str, day_activities in daily_activity_map.items():
    for activity in day_activities:
        activity_signups = signups_df[(signups_df['日期'] == date_str) & 
                                       (signups_df['活动名'] == activity)]
        if len(activity_signups) > 0:
            signup_players = activity_signups['玩家'].unique().tolist()
            num_games = random.randint(2, 6)
            for game in range(num_games):
                table = random.randint(1, 5)
                game_player_count = min(random.randint(4, 8), len(signup_players))
                game_players = random.sample(signup_players, game_player_count)
                game_duration = random.randint(20, 120)
                for player in game_players:
                    results.append({
                        '日期': date_str,
                        '活动名': activity,
                        '玩家': player,
                        '桌号': table,
                        '胜负': random.choice(['胜', '负', '平']),
                        '时长': game_duration
                    })

results_df = pd.DataFrame(results)
results_df.to_csv('data/game_results.csv', index=False, encoding='utf-8-sig')

feedbacks = []
for date_str, day_activities in daily_activity_map.items():
    for activity in day_activities:
        activity_signups = signups_df[(signups_df['日期'] == date_str) & 
                                       (signups_df['活动名'] == activity)]
        if len(activity_signups) > 0:
            signup_players = activity_signups['玩家'].unique().tolist()
            num_feedback = random.randint(3, min(15, len(signup_players)))
            feedback_players = random.sample(signup_players, num_feedback)
            for player in feedback_players:
                feedbacks.append({
                    '日期': date_str,
                    '活动名': activity,
                    '玩家': player,
                    '评分': round(random.uniform(2.5, 5.0), 1),
                    '反馈标签': random.choice(feedback_tags)
                })

feedbacks_df = pd.DataFrame(feedbacks)
feedbacks_df.to_csv('data/feedbacks.csv', index=False, encoding='utf-8-sig')

print('CSV files generated successfully!')
print(f'signups.csv: {len(signups_df)} rows')
print(f'game_results.csv: {len(results_df)} rows')
print(f'feedbacks.csv: {len(feedbacks_df)} rows')

print('\n数据一致性验证:')
signup_keys = set(zip(signups_df['日期'], signups_df['活动名']))
result_keys = set(zip(results_df['日期'], results_df['活动名']))
feedback_keys = set(zip(feedbacks_df['日期'], feedbacks_df['活动名']))
print(f'报名活动组合数: {len(signup_keys)}')
print(f'对局活动组合数: {len(result_keys)}')
print(f'反馈活动组合数: {len(feedback_keys)}')
print(f'所有活动组合一致: {signup_keys == result_keys}')
