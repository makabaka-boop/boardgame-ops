"""核心数据处理函数的单元测试。

覆盖：
- calculate_activity_metrics 指标聚合（参与率使用可配置基数）
- get_high_risk_activities 高风险判定与字段标记
- generate_review_suggestions 风险等级与排序
- _classify_risk_level 风险等级映射
"""
import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import (
    DEFAULT_TOTAL_PLAYER_BASE,
    calculate_activity_metrics,
    compute_risk_indicators,
    get_high_risk_activities,
    generate_review_suggestions,
    _classify_risk_level,
)


def _build_sample_data():
    """构造覆盖三种风险情形（低参与/短时长/低评分）和正常活动的数据。"""
    signups = pd.DataFrame([
        {'日期': '2026-01-01', '活动名': 'A', '玩家': f'P{i:03d}', '桌号': 1}
        for i in range(5)
    ] + [
        {'日期': '2026-01-02', '活动名': 'B', '玩家': f'P{i:03d}', '桌号': 1}
        for i in range(20)
    ] + [
        {'日期': '2026-01-03', '活动名': 'C', '玩家': f'P{i:03d}', '桌号': 1}
        for i in range(20)
    ] + [
        {'日期': '2026-01-04', '活动名': 'D', '玩家': f'P{i:03d}', '桌号': 1}
        for i in range(20)
    ])

    results = pd.DataFrame([
        {'日期': '2026-01-01', '活动名': 'A', '玩家': 'P000', '桌号': 1, '胜负': '胜', '时长': 60},
        {'日期': '2026-01-01', '活动名': 'A', '玩家': 'P001', '桌号': 1, '胜负': '负', '时长': 70},
        {'日期': '2026-01-02', '活动名': 'B', '玩家': 'P000', '桌号': 1, '胜负': '胜', '时长': 30},
        {'日期': '2026-01-02', '活动名': 'B', '玩家': 'P001', '桌号': 1, '胜负': '负', '时长': 35},
        {'日期': '2026-01-02', '活动名': 'B', '玩家': 'P002', '桌号': 1, '胜负': '胜', '时长': 38},
        {'日期': '2026-01-02', '活动名': 'B', '玩家': 'P003', '桌号': 1, '胜负': '负', '时长': 60},
        {'日期': '2026-01-03', '活动名': 'C', '玩家': 'P000', '桌号': 1, '胜负': '胜', '时长': 60},
        {'日期': '2026-01-03', '活动名': 'C', '玩家': 'P001', '桌号': 1, '胜负': '负', '时长': 70},
        {'日期': '2026-01-04', '活动名': 'D', '玩家': 'P000', '桌号': 1, '胜负': '胜', '时长': 80},
        {'日期': '2026-01-04', '活动名': 'D', '玩家': 'P001', '桌号': 1, '胜负': '负', '时长': 75},
    ])

    feedbacks = pd.DataFrame([
        {'日期': '2026-01-01', '活动名': 'A', '玩家': 'P000', '评分': 4.5, '反馈标签': '气氛热烈'},
        {'日期': '2026-01-02', '活动名': 'B', '玩家': 'P000', '评分': 4.0, '反馈标签': '体验极佳'},
        {'日期': '2026-01-03', '活动名': 'C', '玩家': 'P000', '评分': 2.5, '反馈标签': '组织混乱'},
        {'日期': '2026-01-03', '活动名': 'C', '玩家': 'P001', '评分': 2.8, '反馈标签': '建议改进'},
        {'日期': '2026-01-03', '活动名': 'C', '玩家': 'P002', '评分': 3.0, '反馈标签': '规则复杂'},
        {'日期': '2026-01-03', '活动名': 'C', '玩家': 'P003', '评分': 4.0, '反馈标签': '体验极佳'},
        {'日期': '2026-01-04', '活动名': 'D', '玩家': 'P000', '评分': 4.5, '反馈标签': '体验极佳'},
    ])

    for df in (signups, results, feedbacks):
        df['日期'] = pd.to_datetime(df['日期'])

    return signups, results, feedbacks


class CalculateActivityMetricsTest(unittest.TestCase):
    def setUp(self):
        self.signups, self.results, self.feedbacks = _build_sample_data()

    def test_default_participation_uses_default_base(self):
        metrics = calculate_activity_metrics(self.signups, self.results, self.feedbacks)
        row_a = metrics[metrics['活动名'] == 'A'].iloc[0]
        self.assertAlmostEqual(row_a['参与率'], 5 / DEFAULT_TOTAL_PLAYER_BASE)

    def test_custom_total_player_base_changes_participation(self):
        metrics = calculate_activity_metrics(
            self.signups, self.results, self.feedbacks, total_player_base=20
        )
        row_b = metrics[metrics['活动名'] == 'B'].iloc[0]
        self.assertAlmostEqual(row_b['参与率'], 1.0)

    def test_invalid_base_raises(self):
        with self.assertRaises(ValueError):
            calculate_activity_metrics(
                self.signups, self.results, self.feedbacks, total_player_base=0
            )

    def test_aggregation_columns_present(self):
        metrics = calculate_activity_metrics(self.signups, self.results, self.feedbacks)
        expected_cols = {
            '活动名', '日期', '参与人数', '桌数', '平均时长',
            '最短时长', '最长时长', '对局总数',
            '平均评分', '最低评分', '最高评分', '反馈人数', '参与率'
        }
        self.assertTrue(expected_cols.issubset(set(metrics.columns)))


class HighRiskDetectionTest(unittest.TestCase):
    def setUp(self):
        self.signups, self.results, self.feedbacks = _build_sample_data()

    def test_low_participation_triggers_high_risk(self):
        # A：参与人数 5 / 50 = 10% < 30%，应当被识别
        high_risk = get_high_risk_activities(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
        )
        self.assertIn('A', high_risk['活动名'].tolist())

    def test_short_duration_ratio_triggers_high_risk(self):
        # B：4 局中 3 局 < 45min（30,35,38），占比 75% >= 30%
        high_risk = get_high_risk_activities(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.0,
            duration_threshold=45,
            score_threshold=0.0,
        )
        self.assertIn('B', high_risk['活动名'].tolist())

    def test_low_score_ratio_triggers_high_risk(self):
        # C：4 条反馈中 3 条 < 3.5，占比 75% >= 30%
        high_risk = get_high_risk_activities(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.0,
            duration_threshold=0,
            score_threshold=3.5,
        )
        self.assertIn('C', high_risk['活动名'].tolist())

    def test_normal_activity_not_flagged(self):
        # D：参与率正常、时长正常、评分正常，不应被列为高风险
        high_risk = get_high_risk_activities(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
            total_player_base=20,
        )
        self.assertNotIn('D', high_risk['活动名'].tolist())

    def test_risk_indicator_flags_match_thresholds(self):
        indicators = compute_risk_indicators(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
        )
        a_row = indicators[indicators['活动名'] == 'A'].iloc[0]
        self.assertTrue(bool(a_row['参与率不足']))


class ReviewSuggestionTest(unittest.TestCase):
    def setUp(self):
        self.signups, self.results, self.feedbacks = _build_sample_data()

    def test_classify_risk_level(self):
        self.assertEqual(_classify_risk_level(0), '低')
        self.assertEqual(_classify_risk_level(1), '中')
        self.assertEqual(_classify_risk_level(2), '高')
        self.assertEqual(_classify_risk_level(3), '极高')

    def test_review_sorted_by_risk_level_desc(self):
        review = generate_review_suggestions(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
        )
        levels = review['风险等级'].tolist()
        rank_order = {'极高': 0, '高': 1, '中': 2, '低': 3}
        ranks = [rank_order[lv] for lv in levels]
        self.assertEqual(ranks, sorted(ranks))

    def test_review_contains_required_columns(self):
        review = generate_review_suggestions(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
        )
        for col in ('风险等级', '建议动作', '复盘结论'):
            self.assertIn(col, review.columns)

    def test_normal_activity_review_text(self):
        review = generate_review_suggestions(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
            total_player_base=20,
        )
        d_row = review[review['活动名'] == 'D'].iloc[0]
        self.assertEqual(d_row['风险等级'], '低')
        self.assertEqual(d_row['建议动作'], '维持现有运营策略')


if __name__ == '__main__':
    unittest.main()
