import unittest
import pandas as pd
from data_utils import (
    calculate_activity_metrics,
    get_high_risk_activities,
    generate_review_suggestions,
    determine_risk_level,
    get_suggested_actions,
    get_review_conclusion,
    _check_risk_flags,
    TOTAL_REGISTERED_PLAYERS,
    DEFAULT_PARTICIPATION_THRESHOLD,
    DEFAULT_DURATION_THRESHOLD,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_DURATION_RISK_RATIO,
    DEFAULT_SCORE_RISK_RATIO,
)


def _make_test_data():
    signups = pd.DataFrame([
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '狼人杀之夜', '玩家': f'玩家{i:03d}', '桌号': (i % 3) + 1}
        for i in range(1, 6)
    ] + [
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '三国杀争霸', '玩家': f'玩家{i:03d}', '桌号': 1}
        for i in range(1, 21)
    ] + [
        {'日期': pd.Timestamp('2026-01-02'), '活动名': '桌游嘉年华', '玩家': f'玩家{i:03d}', '桌号': (i % 5) + 1}
        for i in range(1, 21)
    ])

    results = pd.DataFrame([
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '狼人杀之夜', '玩家': '玩家001', '桌号': 1, '胜负': '胜', '时长': 60},
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '狼人杀之夜', '玩家': '玩家002', '桌号': 1, '胜负': '负', '时长': 25},
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '狼人杀之夜', '玩家': '玩家003', '桌号': 2, '胜负': '胜', '时长': 30},
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '狼人杀之夜', '玩家': '玩家004', '桌号': 2, '胜负': '平', '时长': 20},
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '狼人杀之夜', '玩家': '玩家005', '桌号': 3, '胜负': '胜', '时长': 70},
    ] + [
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '三国杀争霸', '玩家': f'玩家{i:03d}', '桌号': 1, '胜负': '胜', '时长': 80}
        for i in range(1, 5)
    ] + [
        {'日期': pd.Timestamp('2026-01-02'), '活动名': '桌游嘉年华', '玩家': f'玩家{i:03d}', '桌号': 1, '胜负': '胜', '时长': 70}
        for i in range(1, 6)
    ])

    feedbacks = pd.DataFrame([
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '狼人杀之夜', '玩家': '玩家001', '评分': 4.5, '反馈标签': '气氛热烈'},
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '狼人杀之夜', '玩家': '玩家002', '评分': 4.0, '反馈标签': '等待时间长'},
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '狼人杀之夜', '玩家': '玩家003', '评分': 3.8, '反馈标签': '组织混乱'},
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '狼人杀之夜', '玩家': '玩家004', '评分': 4.2, '反馈标签': '规则复杂'},
    ] + [
        {'日期': pd.Timestamp('2026-01-01'), '活动名': '三国杀争霸', '玩家': f'玩家{i:03d}', '评分': 4.5, '反馈标签': '体验极佳'}
        for i in range(1, 6)
    ] + [
        {'日期': pd.Timestamp('2026-01-02'), '活动名': '桌游嘉年华', '玩家': f'玩家{i:03d}', '评分': 4.0, '反馈标签': '新手友好'}
        for i in range(1, 6)
    ])

    return signups, results, feedbacks


class TestCalculateActivityMetrics(unittest.TestCase):

    def setUp(self):
        self.signups, self.results, self.feedbacks = _make_test_data()

    def test_returns_correct_columns(self):
        metrics = calculate_activity_metrics(self.signups, self.results, self.feedbacks)
        expected_cols = {'活动名', '日期', '参与人数', '桌数', '平均时长', '最短时长',
                         '最长时长', '对局总数', '平均评分', '最低评分', '最高评分', '反馈人数', '参与率'}
        self.assertTrue(expected_cols.issubset(set(metrics.columns)))

    def test_returns_correct_row_count(self):
        metrics = calculate_activity_metrics(self.signups, self.results, self.feedbacks)
        self.assertEqual(len(metrics), 3)

    def test_participation_rate_calculation(self):
        metrics = calculate_activity_metrics(self.signups, self.results, self.feedbacks)
        werewolf = metrics[metrics['活动名'] == '狼人杀之夜'].iloc[0]
        self.assertAlmostEqual(werewolf['参与人数'], 5)
        self.assertAlmostEqual(werewolf['参与率'], 5 / TOTAL_REGISTERED_PLAYERS)

    def test_custom_total_players(self):
        metrics = calculate_activity_metrics(self.signups, self.results, self.feedbacks, total_players=25)
        werewolf = metrics[metrics['活动名'] == '狼人杀之夜'].iloc[0]
        self.assertAlmostEqual(werewolf['参与率'], 5 / 25)

    def test_average_duration(self):
        metrics = calculate_activity_metrics(self.signups, self.results, self.feedbacks)
        sanguosha = metrics[metrics['活动名'] == '三国杀争霸'].iloc[0]
        self.assertAlmostEqual(sanguosha['平均时长'], 80.0, places=1)

    def test_average_score(self):
        metrics = calculate_activity_metrics(self.signups, self.results, self.feedbacks)
        carnival = metrics[metrics['活动名'] == '桌游嘉年华'].iloc[0]
        self.assertAlmostEqual(carnival['平均评分'], 4.0, places=1)

    def test_game_count(self):
        metrics = calculate_activity_metrics(self.signups, self.results, self.feedbacks)
        werewolf = metrics[metrics['活动名'] == '狼人杀之夜'].iloc[0]
        self.assertEqual(werewolf['对局总数'], 5)


class TestRiskLevel(unittest.TestCase):

    def test_zero_flags_is_low(self):
        self.assertEqual(determine_risk_level(0), '低')

    def test_one_flag_is_medium(self):
        self.assertEqual(determine_risk_level(1), '中')

    def test_two_flags_is_high(self):
        self.assertEqual(determine_risk_level(2), '高')

    def test_three_flags_is_extreme(self):
        self.assertEqual(determine_risk_level(3), '极高')


class TestSuggestedActions(unittest.TestCase):

    def test_no_flags_returns_maintain(self):
        result = get_suggested_actions([])
        self.assertEqual(result, '维持现有运营策略')

    def test_participation_flag(self):
        result = get_suggested_actions(['participation'])
        self.assertIn('宣传推广', result)

    def test_duration_flag(self):
        result = get_suggested_actions(['duration'])
        self.assertIn('游戏深度', result)

    def test_score_flag(self):
        result = get_suggested_actions(['score'])
        self.assertIn('体验设计', result)

    def test_multiple_flags_joined(self):
        result = get_suggested_actions(['participation', 'duration', 'score'])
        self.assertIn('；', result)
        self.assertIn('宣传推广', result)
        self.assertIn('游戏深度', result)
        self.assertIn('体验设计', result)


class TestCheckRiskFlags(unittest.TestCase):

    def _make_row(self, participation_rate=0.5, short_ratio=0.1, low_score_ratio=0.1):
        return pd.Series({
            '参与率': participation_rate,
            '短时长占比': short_ratio,
            '低评分占比': low_score_ratio,
        })

    def test_no_flags_when_all_good(self):
        row = self._make_row()
        flags = _check_risk_flags(row, 0.3, 45, 3.5, 0.3, 0.3)
        self.assertEqual(flags, [])

    def test_participation_flag(self):
        row = self._make_row(participation_rate=0.2)
        flags = _check_risk_flags(row, 0.3, 45, 3.5, 0.3, 0.3)
        self.assertIn('participation', flags)

    def test_duration_flag(self):
        row = self._make_row(short_ratio=0.5)
        flags = _check_risk_flags(row, 0.3, 45, 3.5, 0.3, 0.3)
        self.assertIn('duration', flags)

    def test_score_flag(self):
        row = self._make_row(low_score_ratio=0.5)
        flags = _check_risk_flags(row, 0.3, 45, 3.5, 0.3, 0.3)
        self.assertIn('score', flags)

    def test_all_three_flags(self):
        row = self._make_row(participation_rate=0.1, short_ratio=0.6, low_score_ratio=0.6)
        flags = _check_risk_flags(row, 0.3, 45, 3.5, 0.3, 0.3)
        self.assertEqual(len(flags), 3)
        self.assertIn('participation', flags)
        self.assertIn('duration', flags)
        self.assertIn('score', flags)

    def test_threshold_boundary(self):
        row = self._make_row(short_ratio=0.3)
        flags = _check_risk_flags(row, 0.3, 45, 3.5, 0.3, 0.3)
        self.assertIn('duration', flags)


class TestGetHighRiskActivities(unittest.TestCase):

    def setUp(self):
        self.signups, self.results, self.feedbacks = _make_test_data()

    def test_werewolf_is_high_risk_duration(self):
        high_risk = get_high_risk_activities(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.3,
            score_risk_ratio=0.3,
        )
        werewolf_risks = high_risk[high_risk['活动名'] == '狼人杀之夜']
        self.assertEqual(len(werewolf_risks), 1)
        risk_type = werewolf_risks.iloc[0]['风险类型']
        self.assertIn('参与率低', risk_type)
        self.assertIn('时长短', risk_type)

    def test_sanguosha_not_high_risk(self):
        high_risk = get_high_risk_activities(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.3,
            score_risk_ratio=0.3,
        )
        sanguosha_risks = high_risk[high_risk['活动名'] == '三国杀争霸']
        self.assertEqual(len(sanguosha_risks), 0)

    def test_participation_risk_identified(self):
        small_signups = pd.DataFrame([
            {'日期': pd.Timestamp('2026-01-03'), '活动名': '冷门活动', '玩家': f'玩家{i:03d}', '桌号': 1}
            for i in range(1, 6)
        ])
        small_results = pd.DataFrame([
            {'日期': pd.Timestamp('2026-01-03'), '活动名': '冷门活动', '玩家': '玩家001', '桌号': 1, '胜负': '胜', '时长': 60},
        ])
        small_feedbacks = pd.DataFrame([
            {'日期': pd.Timestamp('2026-01-03'), '活动名': '冷门活动', '玩家': '玩家001', '评分': 4.0, '反馈标签': '不错'},
        ])
        high_risk = get_high_risk_activities(
            small_signups, small_results, small_feedbacks,
            participation_threshold=0.5,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.9,
            score_risk_ratio=0.9,
        )
        self.assertEqual(len(high_risk), 1)
        self.assertIn('参与率低', high_risk.iloc[0]['风险类型'])

    def test_default_thresholds_work(self):
        high_risk = get_high_risk_activities(self.signups, self.results, self.feedbacks)
        self.assertIsInstance(high_risk, pd.DataFrame)
        self.assertIn('风险类型', high_risk.columns)

    def test_sorted_by_date_desc(self):
        high_risk = get_high_risk_activities(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.05,
            duration_threshold=30,
            score_threshold=3.0,
            duration_risk_ratio=0.5,
            score_risk_ratio=0.5,
        )
        if len(high_risk) > 1:
            dates = high_risk['日期'].tolist()
            for i in range(len(dates) - 1):
                self.assertGreaterEqual(dates[i], dates[i + 1])


class TestGenerateReviewSuggestions(unittest.TestCase):

    def setUp(self):
        self.signups, self.results, self.feedbacks = _make_test_data()

    def test_returns_all_activities(self):
        review = generate_review_suggestions(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.3,
            score_risk_ratio=0.3,
        )
        self.assertEqual(len(review), 3)

    def test_has_required_columns(self):
        review = generate_review_suggestions(self.signups, self.results, self.feedbacks)
        self.assertIn('风险等级', review.columns)
        self.assertIn('建议动作', review.columns)
        self.assertIn('复盘结论', review.columns)

    def test_risk_levels_correct(self):
        review = generate_review_suggestions(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.3,
            score_risk_ratio=0.3,
        )
        werewolf = review[review['活动名'] == '狼人杀之夜'].iloc[0]
        self.assertEqual(werewolf['风险等级'], '高')

        sanguosha = review[review['活动名'] == '三国杀争霸'].iloc[0]
        self.assertEqual(sanguosha['风险等级'], '低')

    def test_sorted_by_risk_then_date(self):
        review = generate_review_suggestions(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.3,
            score_risk_ratio=0.3,
        )
        risk_order = {'极高': 0, '高': 1, '中': 2, '低': 3}
        risk_values = [risk_order[r] for r in review['风险等级'].tolist()]
        for i in range(len(risk_values) - 1):
            self.assertLessEqual(risk_values[i], risk_values[i + 1])

    def test_low_risk_has_maintain_action(self):
        review = generate_review_suggestions(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.3,
            score_risk_ratio=0.3,
        )
        sanguosha = review[review['活动名'] == '三国杀争霸'].iloc[0]
        self.assertIn('维持现有运营策略', sanguosha['建议动作'])
        self.assertIn('正常', sanguosha['复盘结论'])

    def test_high_risk_has_issues(self):
        review = generate_review_suggestions(
            self.signups, self.results, self.feedbacks,
            participation_threshold=0.3,
            duration_threshold=45,
            score_threshold=3.5,
            duration_risk_ratio=0.3,
            score_risk_ratio=0.3,
        )
        werewolf = review[review['活动名'] == '狼人杀之夜'].iloc[0]
        self.assertIn('报名热度不足', werewolf['复盘结论'])
        self.assertIn('流程节奏偏短', werewolf['复盘结论'])


class TestGetReviewConclusion(unittest.TestCase):

    def _make_int_row(self, **kwargs):
        data = {
            '参与率': 0.5, '短时长对局数': 0, '对局总数': 10,
            '短时长占比': 0.0, '低评分反馈数': 0, '反馈人数': 5, '低评分占比': 0.0
        }
        data.update(kwargs)
        for k in ['短时长对局数', '对局总数', '低评分反馈数', '反馈人数']:
            data[k] = int(data[k])
        return pd.Series(data)

    def test_no_issues_returns_normal(self):
        row = self._make_int_row()
        result = get_review_conclusion(row, [], 0.3, 45, 3.5, 0.3, 0.3)
        self.assertIn('正常', result)

    def test_participation_issue(self):
        row = self._make_int_row(参与率=0.15)
        result = get_review_conclusion(row, ['participation'], 0.3, 45, 3.5, 0.3, 0.3)
        self.assertIn('报名热度不足', result)
        self.assertIn('15%', result)

    def test_duration_issue(self):
        row = self._make_int_row(短时长对局数=4, 对局总数=10, 短时长占比=0.4)
        result = get_review_conclusion(row, ['duration'], 0.3, 45, 3.5, 0.3, 0.3)
        self.assertIn('流程节奏偏短', result)
        self.assertIn('4/10', result)

    def test_score_issue(self):
        row = self._make_int_row(低评分反馈数=3, 反馈人数=5, 低评分占比=0.6)
        result = get_review_conclusion(row, ['score'], 0.3, 45, 3.5, 0.3, 0.3)
        self.assertIn('体验反馈偏低', result)
        self.assertIn('3/5', result)


if __name__ == '__main__':
    unittest.main()
