import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from data_utils import (
    TOTAL_PLAYER_COUNT,
    RISK_LEVEL_ORDER,
    calculate_activity_metrics,
    get_high_risk_activities,
    generate_review_suggestions,
    _classify_risk_level,
    _compute_risk_indicators,
)


def _make_dfs(date_str, activity, signup_players, game_durations, feedback_scores):
    date = pd.to_datetime(date_str)
    signups = pd.DataFrame([
        {'日期': date, '活动名': activity, '玩家': p, '桌号': 1}
        for p in signup_players
    ])
    results = pd.DataFrame([
        {'日期': date, '活动名': activity, '玩家': signup_players[i % len(signup_players)],
         '桌号': 1, '胜负': '胜', '时长': d}
        for i, d in enumerate(game_durations)
    ])
    feedbacks = pd.DataFrame([
        {'日期': date, '活动名': activity, '玩家': signup_players[i % len(signup_players)],
         '评分': s, '反馈标签': '测试'}
        for i, s in enumerate(feedback_scores)
    ])
    return signups, results, feedbacks


class TestCalculateActivityMetrics(unittest.TestCase):

    def test_participation_rate_uses_constant(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '测试活动',
            signup_players=[f'P{i}' for i in range(15)],
            game_durations=[60, 70],
            feedback_scores=[4.0, 4.5]
        )
        metrics = calculate_activity_metrics(signups, results, feedbacks)
        expected_rate = 15 / TOTAL_PLAYER_COUNT
        self.assertAlmostEqual(metrics.iloc[0]['参与率'], expected_rate)

    def test_aggregation_values(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '测试活动',
            signup_players=['P1', 'P2', 'P3', 'P4', 'P5'],
            game_durations=[30, 60, 90],
            feedback_scores=[3.0, 4.0, 5.0]
        )
        metrics = calculate_activity_metrics(signups, results, feedbacks)
        row = metrics.iloc[0]
        self.assertEqual(row['参与人数'], 5)
        self.assertEqual(row['对局总数'], 3)
        self.assertEqual(row['反馈人数'], 3)
        self.assertAlmostEqual(row['平均时长'], 60.0)
        self.assertAlmostEqual(row['平均评分'], 4.0)

    def test_nan_filling_when_no_results(self):
        signups = pd.DataFrame([
            {'日期': pd.to_datetime('2026-01-01'), '活动名': 'A', '玩家': 'P1', '桌号': 1}
        ])
        results = pd.DataFrame(columns=['日期', '活动名', '玩家', '桌号', '胜负', '时长'])
        feedbacks = pd.DataFrame(columns=['日期', '活动名', '玩家', '评分', '反馈标签'])
        metrics = calculate_activity_metrics(signups, results, feedbacks)
        row = metrics.iloc[0]
        self.assertEqual(row['对局总数'], 0)
        self.assertEqual(row['反馈人数'], 0)
        self.assertAlmostEqual(row['平均时长'], 60.0)
        self.assertAlmostEqual(row['平均评分'], 3.5)


class TestHighRiskActivities(unittest.TestCase):

    def test_low_participation_flagged(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '冷门活动',
            signup_players=[f'P{i}' for i in range(5)],
            game_durations=[60, 70, 80],
            feedback_scores=[4.0, 4.5, 5.0]
        )
        high_risk = get_high_risk_activities(
            signups, results, feedbacks,
            participation_threshold=0.3, duration_threshold=45, score_threshold=3.5,
            duration_risk_ratio=0.3, score_risk_ratio=0.3
        )
        self.assertEqual(len(high_risk), 1)
        self.assertIn('参与率低', high_risk.iloc[0]['风险类型'])

    def test_short_duration_flagged(self):
        players = [f'P{i}' for i in range(20)]
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '短局活动',
            signup_players=players,
            game_durations=[20, 25, 30, 80, 90],
            feedback_scores=[4.0, 4.5, 5.0, 4.2, 4.8]
        )
        high_risk = get_high_risk_activities(
            signups, results, feedbacks,
            participation_threshold=0.3, duration_threshold=45, score_threshold=3.5,
            duration_risk_ratio=0.5, score_risk_ratio=0.3
        )
        self.assertEqual(len(high_risk), 1)
        self.assertIn('时长短', high_risk.iloc[0]['风险类型'])

    def test_low_score_flagged(self):
        players = [f'P{i}' for i in range(20)]
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '差评活动',
            signup_players=players,
            game_durations=[60, 70, 80],
            feedback_scores=[2.0, 2.5, 3.0, 4.5]
        )
        high_risk = get_high_risk_activities(
            signups, results, feedbacks,
            participation_threshold=0.3, duration_threshold=45, score_threshold=3.5,
            duration_risk_ratio=0.3, score_risk_ratio=0.5
        )
        self.assertEqual(len(high_risk), 1)
        self.assertIn('评分低', high_risk.iloc[0]['风险类型'])

    def test_safe_activity_not_flagged(self):
        players = [f'P{i}' for i in range(25)]
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '健康活动',
            signup_players=players,
            game_durations=[60, 70, 80, 90],
            feedback_scores=[4.0, 4.2, 4.5, 5.0]
        )
        high_risk = get_high_risk_activities(
            signups, results, feedbacks,
            participation_threshold=0.3, duration_threshold=45, score_threshold=3.5,
            duration_risk_ratio=0.3, score_risk_ratio=0.3
        )
        self.assertEqual(len(high_risk), 0)

    def test_multiple_risk_types(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '问题活动',
            signup_players=['P1', 'P2', 'P3'],
            game_durations=[20, 25, 30],
            feedback_scores=[2.0, 2.5]
        )
        high_risk = get_high_risk_activities(
            signups, results, feedbacks,
            participation_threshold=0.3, duration_threshold=45, score_threshold=3.5,
            duration_risk_ratio=0.3, score_risk_ratio=0.3
        )
        self.assertEqual(len(high_risk), 1)
        risk_type = high_risk.iloc[0]['风险类型']
        self.assertIn('参与率低', risk_type)
        self.assertIn('时长短', risk_type)
        self.assertIn('评分低', risk_type)


class TestReviewSuggestions(unittest.TestCase):

    def _make_row(self, participation_rate=0.5, short_ratio=0.1, low_score_ratio=0.1):
        return pd.Series({
            '参与率': participation_rate,
            '短时长占比': short_ratio,
            '低评分占比': low_score_ratio,
        })

    def test_risk_level_classification(self):
        self.assertEqual(_classify_risk_level(self._make_row(0.5, 0.1, 0.1), 0.3, 0.3, 0.3), '低')
        self.assertEqual(_classify_risk_level(self._make_row(0.2, 0.1, 0.1), 0.3, 0.3, 0.3), '中')
        self.assertEqual(_classify_risk_level(self._make_row(0.2, 0.5, 0.1), 0.3, 0.3, 0.3), '高')
        self.assertEqual(_classify_risk_level(self._make_row(0.2, 0.5, 0.5), 0.3, 0.3, 0.3), '极高')

    def test_sorting_by_risk_then_date(self):
        date1 = pd.to_datetime('2026-03-01')
        date2 = pd.to_datetime('2026-01-01')

        signups = pd.DataFrame([
            {'日期': date1, '活动名': '低风险', '玩家': f'P{i}', '桌号': 1}
            for i in range(25)
        ] + [
            {'日期': date2, '活动名': '高风险', '玩家': f'P{i}', '桌号': 1}
            for i in range(3)
        ])
        results = pd.DataFrame([
            {'日期': date1, '活动名': '低风险', '玩家': f'P{i}', '桌号': 1, '胜负': '胜', '时长': 60}
            for i in range(4)
        ] + [
            {'日期': date2, '活动名': '高风险', '玩家': f'P{i}', '桌号': 1, '胜负': '胜', '时长': 20}
            for i in range(3)
        ])
        feedbacks = pd.DataFrame([
            {'日期': date1, '活动名': '低风险', '玩家': f'P{i}', '评分': 4.5, '反馈标签': '好'}
            for i in range(3)
        ] + [
            {'日期': date2, '活动名': '高风险', '玩家': f'P{i}', '评分': 2.0, '反馈标签': '差'}
            for i in range(2)
        ])

        review = generate_review_suggestions(
            signups, results, feedbacks,
            participation_threshold=0.3, duration_threshold=45, score_threshold=3.5,
            duration_risk_ratio=0.3, score_risk_ratio=0.3
        )
        self.assertEqual(review.iloc[0]['活动名'], '高风险')
        self.assertEqual(review.iloc[0]['风险等级'], '极高')
        self.assertEqual(review.iloc[1]['活动名'], '低风险')
        self.assertEqual(review.iloc[1]['风险等级'], '低')

    def test_suggested_actions_present(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '测试',
            signup_players=['P1', 'P2', 'P3'],
            game_durations=[20, 25, 30],
            feedback_scores=[2.0, 2.5, 3.0]
        )
        review = generate_review_suggestions(
            signups, results, feedbacks,
            participation_threshold=0.3, duration_threshold=45, score_threshold=3.5,
            duration_risk_ratio=0.3, score_risk_ratio=0.3
        )
        actions = review.iloc[0]['建议动作']
        self.assertIn('宣传推广', actions)
        self.assertIn('调整活动节奏', actions)
        self.assertIn('负面反馈', actions)

    def test_safe_activity_default_action(self):
        players = [f'P{i}' for i in range(25)]
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '健康活动',
            signup_players=players,
            game_durations=[60, 70, 80],
            feedback_scores=[4.0, 4.5, 5.0]
        )
        review = generate_review_suggestions(
            signups, results, feedbacks,
            participation_threshold=0.3, duration_threshold=45, score_threshold=3.5,
            duration_risk_ratio=0.3, score_risk_ratio=0.3
        )
        self.assertIn('维持现有运营策略', review.iloc[0]['建议动作'])
        self.assertIn('正常', review.iloc[0]['复盘结论'])

    def test_risk_level_order_constant(self):
        self.assertEqual(RISK_LEVEL_ORDER['极高'], 0)
        self.assertEqual(RISK_LEVEL_ORDER['高'], 1)
        self.assertEqual(RISK_LEVEL_ORDER['中'], 2)
        self.assertEqual(RISK_LEVEL_ORDER['低'], 3)


class TestComputeRiskIndicators(unittest.TestCase):

    def test_ratios_zero_when_no_games_or_feedbacks(self):
        signups = pd.DataFrame([
            {'日期': pd.to_datetime('2026-01-01'), '活动名': 'A', '玩家': 'P1', '桌号': 1}
        ])
        results = pd.DataFrame(columns=['日期', '活动名', '玩家', '桌号', '胜负', '时长'])
        feedbacks = pd.DataFrame(columns=['日期', '活动名', '玩家', '评分', '反馈标签'])
        metrics = calculate_activity_metrics(signups, results, feedbacks)
        risk_df = _compute_risk_indicators(metrics, results, feedbacks, 45, 3.5)
        row = risk_df.iloc[0]
        self.assertEqual(row['短时长对局数'], 0)
        self.assertEqual(row['低评分反馈数'], 0)
        self.assertEqual(row['短时长占比'], 0)
        self.assertEqual(row['低评分占比'], 0)


if __name__ == '__main__':
    unittest.main()
