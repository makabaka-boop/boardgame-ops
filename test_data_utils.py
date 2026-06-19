import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from data_utils import (
    TOTAL_PLAYERS,
    calculate_activity_metrics,
    get_high_risk_activities,
    generate_review_suggestions,
    _determine_risk_level,
    _enrich_risk_indicators,
)


def _make_dfs(date_str, activity, players, durations, scores):
    date = pd.Timestamp(date_str)
    n = len(players)
    signups = pd.DataFrame({
        '日期': [date] * n,
        '活动名': [activity] * n,
        '玩家': players,
        '桌号': [1] * n,
    })
    results = pd.DataFrame({
        '日期': [date] * len(durations),
        '活动名': [activity] * len(durations),
        '玩家': players[:len(durations)],
        '桌号': [1] * len(durations),
        '胜负': ['胜'] * len(durations),
        '时长': durations,
    })
    feedbacks = pd.DataFrame({
        '日期': [date] * len(scores),
        '活动名': [activity] * len(scores),
        '玩家': players[:len(scores)],
        '评分': scores,
        '反馈标签': ['测试'] * len(scores),
    })
    return signups, results, feedbacks


class TestCalculateActivityMetrics(unittest.TestCase):

    def test_participation_rate_uses_total_players_constant(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '测试活动',
            [f'p{i}' for i in range(10)],
            [60, 60],
            [4.0, 4.0],
        )
        metrics = calculate_activity_metrics(signups, results, feedbacks)
        self.assertEqual(len(metrics), 1)
        self.assertAlmostEqual(metrics.iloc[0]['参与率'], 10 / TOTAL_PLAYERS)

    def test_participation_rate_respects_custom_total(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '测试活动',
            [f'p{i}' for i in range(10)],
            [60],
            [4.0],
        )
        metrics = calculate_activity_metrics(signups, results, feedbacks, total_players=20)
        self.assertAlmostEqual(metrics.iloc[0]['参与率'], 0.5)

    def test_aggregation_fields(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '测试活动',
            ['p1', 'p2', 'p3', 'p4'],
            [30, 60, 90],
            [3.0, 5.0],
        )
        metrics = calculate_activity_metrics(signups, results, feedbacks, total_players=10)
        row = metrics.iloc[0]
        self.assertEqual(row['参与人数'], 4)
        self.assertEqual(row['对局总数'], 3)
        self.assertEqual(row['反馈人数'], 2)
        self.assertAlmostEqual(row['平均时长'], 60.0)
        self.assertAlmostEqual(row['平均评分'], 4.0)

    def test_fillna_defaults_when_no_results_or_feedbacks(self):
        signups = pd.DataFrame({
            '日期': [pd.Timestamp('2026-01-01')] * 3,
            '活动名': ['空活动'] * 3,
            '玩家': ['p1', 'p2', 'p3'],
            '桌号': [1, 1, 2],
        })
        results = pd.DataFrame(columns=['日期', '活动名', '玩家', '桌号', '胜负', '时长'])
        feedbacks = pd.DataFrame(columns=['日期', '活动名', '玩家', '评分', '反馈标签'])
        metrics = calculate_activity_metrics(signups, results, feedbacks, total_players=10)
        row = metrics.iloc[0]
        self.assertEqual(row['对局总数'], 0)
        self.assertEqual(row['反馈人数'], 0)
        self.assertAlmostEqual(row['平均时长'], 60)
        self.assertAlmostEqual(row['平均评分'], 3.5)

    def test_multiple_activities_grouped_correctly(self):
        d1, d2 = pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-02')
        signups = pd.DataFrame({
            '日期': [d1, d1, d2, d2, d2],
            '活动名': ['A', 'A', 'A', 'B', 'B'],
            '玩家': ['p1', 'p2', 'p1', 'p2', 'p3'],
            '桌号': [1, 1, 1, 1, 1],
        })
        results = signups.copy()
        results['胜负'] = '胜'
        results['时长'] = 60
        feedbacks = signups.copy()
        feedbacks['评分'] = 4.0
        feedbacks['反馈标签'] = '测试'
        metrics = calculate_activity_metrics(signups, results, feedbacks, total_players=10)
        self.assertEqual(len(metrics), 3)
        a_d1 = metrics[(metrics['活动名'] == 'A') & (metrics['日期'] == d1)].iloc[0]
        a_d2 = metrics[(metrics['活动名'] == 'A') & (metrics['日期'] == d2)].iloc[0]
        b_d2 = metrics[(metrics['活动名'] == 'B') & (metrics['日期'] == d2)].iloc[0]
        self.assertEqual(a_d1['参与人数'], 2)
        self.assertEqual(a_d2['参与人数'], 1)
        self.assertEqual(b_d2['参与人数'], 2)


class TestHighRiskActivities(unittest.TestCase):

    def test_participation_rate_triggers_risk(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '低参与',
            [f'p{i}' for i in range(5)],
            [60] * 5,
            [4.0] * 5,
        )
        high = get_high_risk_activities(
            signups, results, feedbacks,
            participation_threshold=0.20,
            duration_threshold=45,
            score_threshold=3.0,
            duration_risk_ratio=0.3,
            score_risk_ratio=0.3,
        )
        self.assertEqual(len(high), 1)
        self.assertIn('参与率低', high.iloc[0]['风险类型'])

    def test_short_duration_triggers_risk(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '短时长',
            [f'p{i}' for i in range(30)],
            [20, 20, 20, 60, 60],
            [4.0] * 5,
        )
        high = get_high_risk_activities(
            signups, results, feedbacks,
            participation_threshold=0.10,
            duration_threshold=45,
            score_threshold=3.0,
            duration_risk_ratio=0.5,
            score_risk_ratio=0.3,
        )
        self.assertEqual(len(high), 1)
        self.assertIn('时长短', high.iloc[0]['风险类型'])

    def test_low_score_triggers_risk(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '低评分',
            [f'p{i}' for i in range(30)],
            [60] * 5,
            [2.5, 2.8, 4.5, 4.0],
        )
        high = get_high_risk_activities(
            signups, results, feedbacks,
            participation_threshold=0.10,
            duration_threshold=30,
            score_threshold=3.5,
            duration_risk_ratio=0.9,
            score_risk_ratio=0.5,
        )
        self.assertEqual(len(high), 1)
        self.assertIn('评分低', high.iloc[0]['风险类型'])

    def test_healthy_activity_not_flagged(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '健康活动',
            [f'p{i}' for i in range(30)],
            [60] * 5,
            [4.5] * 5,
        )
        high = get_high_risk_activities(
            signups, results, feedbacks,
            participation_threshold=0.20,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.3,
            score_risk_ratio=0.3,
        )
        self.assertEqual(len(high), 0)

    def test_risk_type_string_format(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '综合风险',
            [f'p{i}' for i in range(5)],
            [20, 20, 20, 20, 60],
            [2.0, 2.0, 4.0, 4.0, 4.0],
        )
        high = get_high_risk_activities(
            signups, results, feedbacks,
            participation_threshold=0.20,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.5,
            score_risk_ratio=0.4,
        )
        risk_type = high.iloc[0]['风险类型']
        self.assertIn('参与率低', risk_type)
        self.assertIn('时长短', risk_type)
        self.assertIn('评分低', risk_type)

    def test_sorted_by_date_desc_participation_asc(self):
        d1 = pd.Timestamp('2026-01-01')
        d2 = pd.Timestamp('2026-01-10')
        rows = []
        for date, act, n_p in [(d1, 'A', 2), (d1, 'B', 5), (d2, 'C', 3)]:
            players = [f'p{i}' for i in range(n_p)]
            s = pd.DataFrame({'日期': [date]*n_p, '活动名': [act]*n_p, '玩家': players, '桌号': 1})
            r = s.head(2).copy(); r['胜负'] = '胜'; r['时长'] = 60
            f = s.head(2).copy(); f['评分'] = 4.0; f['反馈标签'] = 'ok'
            rows.append((s, r, f))
        signups = pd.concat([r[0] for r in rows], ignore_index=True)
        results = pd.concat([r[1] for r in rows], ignore_index=True)
        feedbacks = pd.concat([r[2] for r in rows], ignore_index=True)
        high = get_high_risk_activities(
            signups, results, feedbacks,
            participation_threshold=0.30,
            duration_threshold=30,
            score_threshold=3.0,
        )
        dates = high['日期'].tolist()
        self.assertTrue(all(dates[i] >= dates[i+1] for i in range(len(dates)-1)))


class TestRiskLevel(unittest.TestCase):

    def _row(self, participation_rate, short_ratio, low_score_ratio):
        return pd.Series({
            '参与率': participation_rate,
            '短时长占比': short_ratio,
            '低评分占比': low_score_ratio,
        })

    def test_no_flags_is_low(self):
        self.assertEqual(_determine_risk_level(self._row(0.5, 0.0, 0.0), 0.3, 0.3, 0.3), '低')

    def test_one_flag_is_medium(self):
        self.assertEqual(_determine_risk_level(self._row(0.1, 0.0, 0.0), 0.3, 0.3, 0.3), '中')
        self.assertEqual(_determine_risk_level(self._row(0.5, 0.5, 0.0), 0.3, 0.3, 0.3), '中')
        self.assertEqual(_determine_risk_level(self._row(0.5, 0.0, 0.5), 0.3, 0.3, 0.3), '中')

    def test_two_flags_is_high(self):
        self.assertEqual(_determine_risk_level(self._row(0.1, 0.5, 0.0), 0.3, 0.3, 0.3), '高')

    def test_three_flags_is_extreme(self):
        self.assertEqual(_determine_risk_level(self._row(0.1, 0.5, 0.5), 0.3, 0.3, 0.3), '极高')


class TestReviewSuggestionsSorting(unittest.TestCase):

    def test_sorted_by_risk_then_date_desc(self):
        d_old = pd.Timestamp('2026-01-01')
        d_new = pd.Timestamp('2026-01-10')

        def make(date, name, n_p, n_short_dur, n_low_score):
            n_dur = n_short_dur + (0 if n_short_dur == 0 else 1)
            n_fb = n_low_score + (0 if n_low_score == 0 else 1)
            players = [f'p{i}' for i in range(n_p)]
            durations = [20] * n_short_dur + [60] * (n_dur - n_short_dur)
            scores = [2.0] * n_low_score + [4.0] * (n_fb - n_low_score)
            r_players = (players * ((n_dur // len(players)) + 1))[:n_dur]
            f_players = (players * ((n_fb // len(players)) + 1))[:n_fb]
            s = pd.DataFrame({'日期': [date]*n_p, '活动名': [name]*n_p, '玩家': players, '桌号': 1})
            r = pd.DataFrame({
                '日期': [date]*n_dur, '活动名': [name]*n_dur,
                '玩家': r_players, '桌号': 1, '胜负': '胜', '时长': durations,
            })
            f = pd.DataFrame({
                '日期': [date]*n_fb, '活动名': [name]*n_fb,
                '玩家': f_players, '评分': scores, '反馈标签': 'x',
            })
            return s, r, f

        extreme = make(d_new, '极高', n_p=5, n_short_dur=4, n_low_score=4)
        high_r = make(d_old, '高', n_p=5, n_short_dur=4, n_low_score=0)
        medium = make(d_new, '中', n_p=5, n_short_dur=0, n_low_score=0)
        low = make(d_new, '低', n_p=30, n_short_dur=0, n_low_score=0)

        signups = pd.concat([x[0] for x in [extreme, high_r, medium, low]], ignore_index=True)
        results = pd.concat([x[1] for x in [extreme, high_r, medium, low]], ignore_index=True)
        feedbacks = pd.concat([x[2] for x in [extreme, high_r, medium, low]], ignore_index=True)

        review = generate_review_suggestions(
            signups, results, feedbacks,
            participation_threshold=0.20,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.3,
            score_risk_ratio=0.3,
        )

        risk_order = {'极高': 0, '高': 1, '中': 2, '低': 3}
        risk_ranks = review['风险等级'].map(risk_order).tolist()
        self.assertTrue(all(risk_ranks[i] <= risk_ranks[i+1] for i in range(len(risk_ranks)-1)))

        levels = review['风险等级'].tolist()
        self.assertEqual(levels[0], '极高')
        self.assertEqual(levels[-1], '低')

    def test_review_conclusion_for_healthy(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '健康',
            [f'p{i}' for i in range(30)],
            [60] * 5,
            [4.5] * 5,
        )
        review = generate_review_suggestions(
            signups, results, feedbacks,
            participation_threshold=0.20,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.3,
            score_risk_ratio=0.3,
        )
        self.assertEqual(review.iloc[0]['风险等级'], '低')
        self.assertIn('正常', review.iloc[0]['复盘结论'])
        self.assertIn('维持', review.iloc[0]['建议动作'])

    def test_suggested_actions_match_flags(self):
        signups, results, feedbacks = _make_dfs(
            '2026-01-01', '多风险',
            [f'p{i}' for i in range(5)],
            [20, 20, 20, 20, 60],
            [2.0, 2.0, 4.0, 4.0, 4.0],
        )
        review = generate_review_suggestions(
            signups, results, feedbacks,
            participation_threshold=0.20,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.5,
            score_risk_ratio=0.4,
        )
        row = review.iloc[0]
        self.assertIn('宣传推广', row['建议动作'])
        self.assertIn('调整活动节奏', row['建议动作'])
        self.assertIn('负面反馈', row['建议动作'])
        self.assertEqual(row['风险等级'], '极高')


class TestEnrichRiskIndicators(unittest.TestCase):

    def test_zero_games_zero_ratio(self):
        signups = pd.DataFrame({
            '日期': [pd.Timestamp('2026-01-01')] * 5,
            '活动名': ['X'] * 5,
            '玩家': [f'p{i}' for i in range(5)],
            '桌号': 1,
        })
        results = pd.DataFrame(columns=['日期', '活动名', '玩家', '桌号', '胜负', '时长'])
        feedbacks = pd.DataFrame(columns=['日期', '活动名', '玩家', '评分', '反馈标签'])
        metrics = calculate_activity_metrics(signups, results, feedbacks, total_players=10)
        enriched = _enrich_risk_indicators(metrics, results, feedbacks, 45, 3.5)
        self.assertEqual(enriched.iloc[0]['短时长对局数'], 0)
        self.assertEqual(enriched.iloc[0]['短时长占比'], 0)
        self.assertEqual(enriched.iloc[0]['低评分反馈数'], 0)
        self.assertEqual(enriched.iloc[0]['低评分占比'], 0)


if __name__ == '__main__':
    unittest.main()
