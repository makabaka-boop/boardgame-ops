import pandas as pd
import os
from datetime import datetime

DATA_DIR = 'data'

DEFAULT_TOTAL_PLAYER_BASE = 50
DEFAULT_DURATION_RISK_RATIO = 0.3
DEFAULT_SCORE_RISK_RATIO = 0.3

DEFAULT_FILL_DURATION = 60
DEFAULT_FILL_SCORE = 3.5

RISK_LEVEL_ORDER = {'极高': 0, '高': 1, '中': 2, '低': 3}


def load_data():
    signups_path = os.path.join(DATA_DIR, 'signups.csv')
    results_path = os.path.join(DATA_DIR, 'game_results.csv')
    feedbacks_path = os.path.join(DATA_DIR, 'feedbacks.csv')

    signups_df = pd.read_csv(signups_path, encoding='utf-8-sig')
    results_df = pd.read_csv(results_path, encoding='utf-8-sig')
    feedbacks_df = pd.read_csv(feedbacks_path, encoding='utf-8-sig')

    signups_df['日期'] = pd.to_datetime(signups_df['日期'])
    results_df['日期'] = pd.to_datetime(results_df['日期'])
    feedbacks_df['日期'] = pd.to_datetime(feedbacks_df['日期'])

    return signups_df, results_df, feedbacks_df


def calculate_activity_metrics(signups_df, results_df, feedbacks_df,
                               total_player_base=DEFAULT_TOTAL_PLAYER_BASE):
    activity_stats = signups_df.groupby(['活动名', '日期']).agg(
        参与人数=('玩家', 'nunique'),
        桌数=('桌号', 'nunique')
    ).reset_index()

    duration_stats = results_df.groupby(['活动名', '日期']).agg(
        平均时长=('时长', 'mean'),
        最短时长=('时长', 'min'),
        最长时长=('时长', 'max'),
        对局总数=('时长', 'count')
    ).reset_index()

    score_stats = feedbacks_df.groupby(['活动名', '日期']).agg(
        平均评分=('评分', 'mean'),
        最低评分=('评分', 'min'),
        最高评分=('评分', 'max'),
        反馈人数=('玩家', 'nunique')
    ).reset_index()

    merged = activity_stats.merge(duration_stats, on=['活动名', '日期'], how='left')
    merged = merged.merge(score_stats, on=['活动名', '日期'], how='left')

    if total_player_base <= 0:
        raise ValueError('total_player_base 必须为正数')

    merged['参与率'] = merged['参与人数'] / total_player_base
    merged['平均时长'] = merged['平均时长'].fillna(DEFAULT_FILL_DURATION)
    merged['最短时长'] = merged['最短时长'].fillna(DEFAULT_FILL_DURATION)
    merged['最长时长'] = merged['最长时长'].fillna(DEFAULT_FILL_DURATION)
    merged['对局总数'] = merged['对局总数'].fillna(0).astype(int)
    merged['平均评分'] = merged['平均评分'].fillna(DEFAULT_FILL_SCORE)
    merged['最低评分'] = merged['最低评分'].fillna(DEFAULT_FILL_SCORE)
    merged['最高评分'] = merged['最高评分'].fillna(DEFAULT_FILL_SCORE)
    merged['反馈人数'] = merged['反馈人数'].fillna(0).astype(int)

    return merged


def _safe_ratio(numerator, denominator):
    if denominator and denominator > 0:
        return numerator / denominator
    return 0


def compute_risk_indicators(signups_df, results_df, feedbacks_df,
                            participation_threshold, duration_threshold,
                            score_threshold,
                            duration_risk_ratio=DEFAULT_DURATION_RISK_RATIO,
                            score_risk_ratio=DEFAULT_SCORE_RISK_RATIO,
                            total_player_base=DEFAULT_TOTAL_PLAYER_BASE):
    """计算每场活动的风险相关指标，统一供高风险判定与复盘建议使用。

    返回的 DataFrame 在 calculate_activity_metrics 基础上额外包含：
    - 短时长对局数 / 低评分反馈数
    - 短时长占比 / 低评分占比
    - 参与率不足 / 短时长超阈 / 低评分超阈 三个布尔标记
    - 风险标记数（命中风险维度数量）
    """
    metrics_df = calculate_activity_metrics(
        signups_df, results_df, feedbacks_df, total_player_base=total_player_base
    )

    short_duration_counts = (
        results_df[results_df['时长'] < duration_threshold]
        .groupby(['活动名', '日期'])
        .size()
        .reset_index(name='短时长对局数')
    )

    low_score_counts = (
        feedbacks_df[feedbacks_df['评分'] < score_threshold]
        .groupby(['活动名', '日期'])
        .size()
        .reset_index(name='低评分反馈数')
    )

    merged = metrics_df.merge(short_duration_counts, on=['活动名', '日期'], how='left')
    merged = merged.merge(low_score_counts, on=['活动名', '日期'], how='left')

    merged['短时长对局数'] = merged['短时长对局数'].fillna(0).astype(int)
    merged['低评分反馈数'] = merged['低评分反馈数'].fillna(0).astype(int)

    merged['短时长占比'] = merged.apply(
        lambda x: _safe_ratio(x['短时长对局数'], x['对局总数']), axis=1
    )
    merged['低评分占比'] = merged.apply(
        lambda x: _safe_ratio(x['低评分反馈数'], x['反馈人数']), axis=1
    )

    merged['参与率不足'] = merged['参与率'] < participation_threshold
    merged['短时长超阈'] = merged['短时长占比'] >= duration_risk_ratio
    merged['低评分超阈'] = merged['低评分占比'] >= score_risk_ratio
    merged['风险标记数'] = (
        merged['参与率不足'].astype(int)
        + merged['短时长超阈'].astype(int)
        + merged['低评分超阈'].astype(int)
    )

    return merged


def _classify_risk_level(flags):
    if flags == 0:
        return '低'
    if flags == 1:
        return '中'
    if flags == 2:
        return '高'
    return '极高'


def get_high_risk_activities(signups_df, results_df, feedbacks_df,
                             participation_threshold, duration_threshold, score_threshold,
                             duration_risk_ratio=DEFAULT_DURATION_RISK_RATIO,
                             score_risk_ratio=DEFAULT_SCORE_RISK_RATIO,
                             total_player_base=DEFAULT_TOTAL_PLAYER_BASE):
    indicators = compute_risk_indicators(
        signups_df, results_df, feedbacks_df,
        participation_threshold, duration_threshold, score_threshold,
        duration_risk_ratio, score_risk_ratio, total_player_base
    )

    high_risk = indicators[indicators['风险标记数'] > 0].copy()

    high_risk['风险类型'] = ''
    participation_mask = high_risk['参与率不足']
    high_risk.loc[participation_mask, '风险类型'] += high_risk.loc[participation_mask, '参与率'].map(
        lambda x: f'参与率低({x*100:.0f}%); '
    )

    duration_risk_mask = high_risk['短时长超阈']
    high_risk.loc[duration_risk_mask, '风险类型'] += high_risk.loc[duration_risk_mask, :].apply(
        lambda x: f'时长短({x["短时长对局数"]}/{x["对局总数"]}局<{duration_threshold}分); ',
        axis=1
    )

    score_risk_mask = high_risk['低评分超阈']
    high_risk.loc[score_risk_mask, '风险类型'] += high_risk.loc[score_risk_mask, :].apply(
        lambda x: f'评分低({x["低评分反馈数"]}/{x["反馈人数"]}人<{score_threshold}分); ',
        axis=1
    )

    high_risk['风险类型'] = high_risk['风险类型'].str.rstrip('; ')

    return high_risk.sort_values(['日期', '参与率'], ascending=[False, True])


def get_short_duration_games(results_df, duration_threshold):
    return results_df[results_df['时长'] < duration_threshold].sort_values('时长')


def get_low_score_feedbacks(feedbacks_df, score_threshold):
    return feedbacks_df[feedbacks_df['评分'] < score_threshold].sort_values('评分')


def get_activity_popularity(signups_df):
    return signups_df.groupby('活动名')['玩家'].nunique().sort_values(ascending=False)


def get_player_retention(signups_df):
    player_dates = signups_df.groupby('玩家')['日期'].agg(['min', 'max', 'count']).reset_index()
    player_dates.columns = ['玩家', '首次参与', '末次参与', '参与次数']
    player_dates['活跃天数'] = (player_dates['末次参与'] - player_dates['首次参与']).dt.days + 1
    return player_dates


def get_duration_distribution(results_df):
    return results_df['时长']


def get_feedback_rankings(feedbacks_df):
    tag_counts = feedbacks_df['反馈标签'].value_counts()
    return tag_counts


def generate_review_suggestions(signups_df, results_df, feedbacks_df,
                                participation_threshold, duration_threshold,
                                score_threshold,
                                duration_risk_ratio=DEFAULT_DURATION_RISK_RATIO,
                                score_risk_ratio=DEFAULT_SCORE_RISK_RATIO,
                                total_player_base=DEFAULT_TOTAL_PLAYER_BASE):
    merged = compute_risk_indicators(
        signups_df, results_df, feedbacks_df,
        participation_threshold, duration_threshold, score_threshold,
        duration_risk_ratio, score_risk_ratio, total_player_base
    )

    def _suggested_action(row):
        actions = []
        if row['参与率不足']:
            actions.append('加强宣传推广，优化报名渠道')
        if row['短时长超阈']:
            actions.append('调整活动节奏，增加游戏深度')
        if row['低评分超阈']:
            actions.append('收集负面反馈细节，改善体验设计')
        if not actions:
            actions.append('维持现有运营策略')
        return '；'.join(actions)

    def _review_conclusion(row):
        issues = []
        if row['参与率不足']:
            issues.append(
                f"报名热度不足（参与率仅{row['参与率']*100:.0f}%，"
                f"低于{participation_threshold*100:.0f}%阈值）"
            )
        if row['短时长超阈']:
            issues.append(
                f"流程节奏偏短（{row['短时长对局数']}/{row['对局总数']}局"
                f"低于{duration_threshold}分钟，占比{row['短时长占比']*100:.0f}%）"
            )
        if row['低评分超阈']:
            issues.append(
                f"体验反馈偏低（{row['低评分反馈数']}/{row['反馈人数']}人"
                f"低于{score_threshold}分，占比{row['低评分占比']*100:.0f}%）"
            )
        if not issues:
            return '活动各项指标正常，整体表现良好'
        return '；'.join(issues)

    merged['风险等级'] = merged['风险标记数'].map(_classify_risk_level)
    merged['建议动作'] = merged.apply(_suggested_action, axis=1)
    merged['复盘结论'] = merged.apply(_review_conclusion, axis=1)

    merged['_risk_sort'] = merged['风险等级'].map(RISK_LEVEL_ORDER)
    merged = merged.sort_values(['_risk_sort', '日期'], ascending=[True, False])
    merged = merged.drop(columns=['_risk_sort'])

    return merged


def generate_html_report(metrics_df, high_risk_df, output_path='report.html'):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>桌游社活动日报 - {datetime.now().strftime('%Y-%m-%d')}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .risk {{ color: #e74c3c; font-weight: bold; }}
            .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>🎲 桌游社活动日报</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary">
            <h2>📊 活动概览</h2>
            <p>统计活动总数: <strong>{len(metrics_df)}</strong> 场</p>
            <p>高风险活动数: <strong class="risk">{len(high_risk_df)}</strong> 场</p>
            <p>平均参与率: <strong>{metrics_df['参与率'].mean()*100:.1f}%</strong></p>
            <p>平均评分: <strong>{metrics_df['平均评分'].mean():.2f}</strong></p>
        </div>

        <h2>⚠️ 高风险活动清单</h2>
        {high_risk_df.to_html(index=False, classes='risk-table') if len(high_risk_df) > 0 else '<p>暂无高风险活动</p>'}

        <h2>📋 所有活动统计</h2>
        {metrics_df.to_html(index=False)}
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return output_path
