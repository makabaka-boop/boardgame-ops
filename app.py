import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_utils import (
    DEFAULT_TOTAL_PLAYER_BASE,
    load_data,
    calculate_activity_metrics,
    get_high_risk_activities,
    get_short_duration_games,
    get_low_score_feedbacks,
    get_activity_popularity,
    get_player_retention,
    get_duration_distribution,
    get_feedback_rankings,
    generate_review_suggestions,
)

RISK_EMOJI = {'极高': '🔴', '高': '🟠', '中': '🟡', '低': '🟢'}


def configure_page():
    st.set_page_config(
        page_title="桌游社活动参数模拟与对比工具",
        page_icon="🎲",
        layout="wide"
    )
    st.title("🎲 桌游社活动参与与复盘分析工具")
    st.markdown("---")


def render_sidebar():
    """渲染侧边栏阈值配置，返回所有阈值参数。"""
    st.sidebar.header("⚙️ 预警阈值设置")
    st.sidebar.markdown("拖动滑块调整预警阈值，实时更新高风险活动清单")

    participation_threshold = st.sidebar.slider(
        "参与率预警阈值 (%)",
        min_value=10, max_value=80, value=30, step=5, format="%d%%"
    ) / 100

    duration_threshold = st.sidebar.slider(
        "对局时长预警阈值 (分钟)",
        min_value=20, max_value=100, value=45, step=5
    )

    score_threshold = st.sidebar.slider(
        "反馈评分预警阈值",
        min_value=2.0, max_value=4.5, value=3.5, step=0.1
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("风险比例设置")
    duration_risk_ratio = st.sidebar.slider(
        "短对局风险比例",
        min_value=10, max_value=80, value=30, step=5, format="%d%%"
    ) / 100

    score_risk_ratio = st.sidebar.slider(
        "低评分风险比例",
        min_value=10, max_value=80, value=30, step=5, format="%d%%"
    ) / 100

    st.sidebar.markdown("---")
    st.sidebar.subheader("活动池设置")
    total_player_base = st.sidebar.number_input(
        "活跃玩家基数（用于计算参与率）",
        min_value=1, max_value=1000,
        value=DEFAULT_TOTAL_PLAYER_BASE, step=1
    )

    return {
        'participation_threshold': participation_threshold,
        'duration_threshold': duration_threshold,
        'score_threshold': score_threshold,
        'duration_risk_ratio': duration_risk_ratio,
        'score_risk_ratio': score_risk_ratio,
        'total_player_base': total_player_base,
    }


def render_overview_metrics(metrics_df, high_risk_df, signups_df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 总活动场次", f"{len(metrics_df)}场")
    with col2:
        st.metric("⚠️ 高风险活动", f"{len(high_risk_df)}场")
    with col3:
        st.metric("👥 活跃玩家", f"{signups_df['玩家'].nunique()}人")
    with col4:
        st.metric("⭐ 平均评分", f"{metrics_df['平均评分'].mean():.2f}")
    st.markdown("---")


def render_popularity_tab(signups_df):
    st.subheader("活动热度排行")
    activity_popularity = get_activity_popularity(signups_df)

    fig_popularity = px.bar(
        x=activity_popularity.values,
        y=activity_popularity.index,
        orientation='h',
        title="各活动参与人数排行",
        labels={'x': '参与人数', 'y': '活动名称'},
        color=activity_popularity.values,
        color_continuous_scale='Viridis'
    )
    fig_popularity.update_layout(height=500)
    st.plotly_chart(fig_popularity, use_container_width=True)

    st.subheader("每日活动参与趋势")
    daily_participation = signups_df.groupby('日期')['玩家'].nunique().reset_index()
    fig_trend = px.line(
        daily_participation,
        x='日期', y='玩家',
        title="每日活跃玩家数趋势",
        labels={'玩家': '活跃玩家数'}
    )
    st.plotly_chart(fig_trend, use_container_width=True)


def render_retention_tab(signups_df):
    st.subheader("玩家留存分析")
    player_retention = get_player_retention(signups_df)

    col_ret1, col_ret2 = st.columns(2)
    with col_ret1:
        fig_retention = px.histogram(
            player_retention,
            x='参与次数',
            title="玩家参与次数分布",
            nbins=20,
            color_discrete_sequence=['#3498db']
        )
        st.plotly_chart(fig_retention, use_container_width=True)

    with col_ret2:
        retention_summary = pd.cut(
            player_retention['参与次数'],
            bins=[0, 1, 3, 5, 10, float('inf')],
            labels=['1次', '2-3次', '4-5次', '6-10次', '10次以上']
        ).value_counts()

        fig_pie = px.pie(
            values=retention_summary.values,
            names=retention_summary.index,
            title="玩家活跃度分布",
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Top 10 活跃玩家")
    top_players = player_retention.nlargest(10, '参与次数')
    st.dataframe(
        top_players[['玩家', '参与次数', '首次参与', '末次参与']],
        use_container_width=True
    )


def render_duration_tab(results_df, duration_threshold, short_games_df):
    st.subheader("对局时长分布")
    duration_data = get_duration_distribution(results_df)

    col_dur1, col_dur2 = st.columns(2)
    with col_dur1:
        results_df['时长状态'] = results_df['时长'].apply(
            lambda x: f'<{duration_threshold}分钟' if x < duration_threshold else f'>={duration_threshold}分钟'
        )
        fig_duration = px.histogram(
            results_df,
            x='时长',
            title="对局时长分布直方图（红色为低于阈值）",
            labels={'x': '时长(分钟)', 'y': '对局数'},
            nbins=20,
            color='时长状态',
            color_discrete_map={
                f'<{duration_threshold}分钟': '#e74c3c',
                f'>={duration_threshold}分钟': '#2ecc71'
            }
        )
        fig_duration.add_vline(
            x=duration_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"预警线: {duration_threshold}分钟"
        )
        st.plotly_chart(fig_duration, use_container_width=True)

    with col_dur2:
        fig_box = px.box(
            results_df,
            x='活动名', y='时长',
            title="各活动对局时长箱线图",
            labels={'时长': '时长(分钟)'}
        )
        fig_box.add_hline(
            y=duration_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"预警线: {duration_threshold}分钟"
        )
        fig_box.update_layout(xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig_box, use_container_width=True)

    st.subheader(f"⏰ 低于阈值的对局详情（共 {len(short_games_df)} 局）")
    st.dataframe(
        short_games_df[['日期', '活动名', '玩家', '桌号', '胜负', '时长']],
        use_container_width=True
    )

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("平均对局时长", f"{duration_data.mean():.1f} 分钟")
    with col_m2:
        st.metric("最短对局时长", f"{duration_data.min()} 分钟")
    with col_m3:
        st.metric(
            "低于阈值对局数",
            f"{len(short_games_df)} 局 ({len(short_games_df)/len(results_df)*100:.1f}%)"
        )


def render_feedback_tab(feedbacks_df, score_threshold, low_score_df):
    st.subheader("反馈问题排行")
    feedback_rank = get_feedback_rankings(feedbacks_df)

    col_fb1, col_fb2 = st.columns(2)
    with col_fb1:
        fig_feedback = px.bar(
            x=feedback_rank.values,
            y=feedback_rank.index,
            orientation='h',
            title="反馈标签分布",
            labels={'x': '出现次数', 'y': '反馈标签'},
            color=feedback_rank.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig_feedback.update_layout(height=400)
        st.plotly_chart(fig_feedback, use_container_width=True)

    with col_fb2:
        avg_score_by_tag = feedbacks_df.groupby('反馈标签')['评分'].mean().sort_values()
        fig_score = px.bar(
            x=avg_score_by_tag.values,
            y=avg_score_by_tag.index,
            orientation='h',
            title="各反馈标签平均评分",
            labels={'x': '平均评分', 'y': '反馈标签'},
            color=avg_score_by_tag.values,
            color_continuous_scale='RdYlGn',
            range_color=[2, 5]
        )
        fig_score.update_layout(height=400)
        st.plotly_chart(fig_score, use_container_width=True)

    st.subheader(f"⭐ 真实反馈评分散点图（共 {len(feedbacks_df)} 条反馈）")
    feedbacks_df['评分状态'] = feedbacks_df['评分'].apply(
        lambda x: f'<{score_threshold}分' if x < score_threshold else f'>={score_threshold}分'
    )
    fig_score_scatter = px.scatter(
        feedbacks_df,
        x='日期', y='评分',
        color='评分状态',
        size='评分',
        hover_data=['活动名', '玩家', '反馈标签'],
        title="每条反馈的真实评分（红色为低于阈值）",
        color_discrete_map={
            f'<{score_threshold}分': '#e74c3c',
            f'>={score_threshold}分': '#2ecc71'
        }
    )
    fig_score_scatter.add_hline(
        y=score_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"预警线: {score_threshold}分"
    )
    fig_score_scatter.update_layout(height=400)
    st.plotly_chart(fig_score_scatter, use_container_width=True)

    st.subheader(f"📝 低于阈值的反馈详情（共 {len(low_score_df)} 条）")
    st.dataframe(
        low_score_df[['日期', '活动名', '玩家', '评分', '反馈标签']],
        use_container_width=True
    )


def render_risk_tab(metrics_df, high_risk_df, results_df, feedbacks_df, thresholds):
    participation_threshold = thresholds['participation_threshold']
    duration_threshold = thresholds['duration_threshold']
    score_threshold = thresholds['score_threshold']
    duration_risk_ratio = thresholds['duration_risk_ratio']
    score_risk_ratio = thresholds['score_risk_ratio']

    st.subheader("⚠️ 高风险活动预警")
    st.info(f"""
    当前预警设置：
    - 参与率 < {participation_threshold*100:.0f}%
    - 短时长对局占比 ≥ {duration_risk_ratio*100:.0f}%（单局时长 < {duration_threshold}分钟）
    - 低评分反馈占比 ≥ {score_risk_ratio*100:.0f}%（单条评分 < {score_threshold}分）
    """)

    if len(high_risk_df) > 0:
        display_cols = ['日期', '活动名', '参与人数', '参与率',
                        '对局总数', '短时长对局数', '短时长占比',
                        '反馈人数', '低评分反馈数', '低评分占比', '风险类型']
        st.dataframe(
            high_risk_df[display_cols].style.format({
                '参与率': '{:.1%}',
                '短时长占比': '{:.1%}',
                '低评分占比': '{:.1%}'
            }),
            use_container_width=True,
            height=400
        )
    else:
        st.success("🎉 暂无高风险活动！所有活动指标均在正常范围内")

    st.subheader("参与率趋势与预警线")
    fig_participation = go.Figure()
    fig_participation.add_trace(go.Scatter(
        x=metrics_df['日期'],
        y=metrics_df['参与率'] * 100,
        mode='markers',
        name='活动参与率',
        marker=dict(
            color=metrics_df['参与率'].apply(
                lambda x: 'red' if x < participation_threshold else 'green'
            ),
            size=10
        ),
        text=metrics_df['活动名']
    ))
    fig_participation.add_hline(
        y=participation_threshold * 100,
        line_dash="dash",
        line_color="red",
        annotation_text=f"预警线: {participation_threshold*100:.0f}%"
    )
    fig_participation.update_layout(
        title="参与率散点图与预警阈值",
        yaxis_title="参与率 (%)",
        height=400
    )
    st.plotly_chart(fig_participation, use_container_width=True)

    col_th1, col_th2 = st.columns(2)
    with col_th1:
        st.subheader("真实对局时长散点图")
        results_df['高风险'] = results_df['时长'] < duration_threshold
        fig_duration_real = px.scatter(
            results_df,
            x='日期', y='时长',
            color='高风险',
            hover_data=['活动名', '玩家', '胜负'],
            title="每局真实时长（红点为低于阈值）",
            color_discrete_map={True: 'red', False: 'green'}
        )
        fig_duration_real.add_hline(
            y=duration_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"预警线: {duration_threshold}分钟"
        )
        fig_duration_real.update_layout(height=350)
        st.plotly_chart(fig_duration_real, use_container_width=True)

    with col_th2:
        st.subheader("真实反馈评分散点图")
        feedbacks_df['高风险'] = feedbacks_df['评分'] < score_threshold
        fig_score_real = px.scatter(
            feedbacks_df,
            x='日期', y='评分',
            color='高风险',
            hover_data=['活动名', '玩家', '反馈标签'],
            title="每条真实反馈评分（红点为低于阈值）",
            color_discrete_map={True: 'red', False: 'green'}
        )
        fig_score_real.add_hline(
            y=score_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"预警线: {score_threshold}分"
        )
        fig_score_real.update_layout(height=350)
        st.plotly_chart(fig_score_real, use_container_width=True)


def _render_review_high_risk_table(high_risk_review):
    st.markdown("#### 🔴 高风险活动列表")
    if len(high_risk_review) == 0:
        st.success("🎉 当前筛选条件下无高风险活动！")
        return

    display_cols = ['日期', '活动名', '风险等级', '建议动作',
                    '参与人数', '参与率',
                    '对局总数', '短时长对局数', '短时长占比',
                    '反馈人数', '低评分反馈数', '低评分占比']
    styled_df = high_risk_review[display_cols].copy()
    styled_df['风险等级'] = styled_df['风险等级'].apply(
        lambda x: f"{RISK_EMOJI.get(x, '')} {x}"
    )
    st.dataframe(
        styled_df.style.format({
            '参与率': '{:.1%}',
            '短时长占比': '{:.1%}',
            '低评分占比': '{:.1%}'
        }),
        use_container_width=True,
        height=350
    )


def _render_review_detail(filtered_df, thresholds):
    participation_threshold = thresholds['participation_threshold']
    duration_threshold = thresholds['duration_threshold']
    score_threshold = thresholds['score_threshold']
    duration_risk_ratio = thresholds['duration_risk_ratio']
    score_risk_ratio = thresholds['score_risk_ratio']

    st.markdown("---")
    st.markdown("#### 📝 活动复盘详情")
    review_options = filtered_df.apply(
        lambda r: f"{r['日期'].strftime('%Y-%m-%d')} - {r['活动名']}（风险：{r['风险等级']}）",
        axis=1
    ).tolist()
    selected_idx = st.selectbox(
        "选择活动查看复盘详情",
        options=range(len(review_options)),
        format_func=lambda i: review_options[i],
        key="review_detail_select"
    )

    selected_row = filtered_df.iloc[selected_idx]

    st.markdown(
        f"**{RISK_EMOJI.get(selected_row['风险等级'], '')} 风险等级：{selected_row['风险等级']}**"
    )
    st.markdown(f"**复盘结论：**{selected_row['复盘结论']}")
    st.markdown(f"**建议动作：**{selected_row['建议动作']}")

    st.markdown("---")
    st.markdown("##### 关键指标摘要")
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1:
        st.metric(
            "参与率", f"{selected_row['参与率']:.1%}",
            delta=f"阈值 {participation_threshold*100:.0f}%",
            delta_color="inverse" if selected_row['参与率'] < participation_threshold else "normal"
        )
    with kc2:
        st.metric(
            "短时长占比", f"{selected_row['短时长占比']:.1%}",
            delta=f"阈值 {duration_risk_ratio*100:.0f}%",
            delta_color="inverse" if selected_row['短时长占比'] >= duration_risk_ratio else "normal"
        )
    with kc3:
        st.metric(
            "低评分占比", f"{selected_row['低评分占比']:.1%}",
            delta=f"阈值 {score_risk_ratio*100:.0f}%",
            delta_color="inverse" if selected_row['低评分占比'] >= score_risk_ratio else "normal"
        )
    with kc4:
        st.metric(
            "平均评分", f"{selected_row['平均评分']:.2f}",
            delta=f"阈值 {score_threshold}分",
            delta_color="inverse" if selected_row['平均评分'] < score_threshold else "normal"
        )

    kc5, kc6, kc7, kc8 = st.columns(4)
    with kc5:
        st.metric("参与人数", f"{int(selected_row['参与人数'])}人")
    with kc6:
        st.metric("对局总数", f"{int(selected_row['对局总数'])}局")
    with kc7:
        st.metric("平均时长", f"{selected_row['平均时长']:.1f}分钟")
    with kc8:
        st.metric("反馈人数", f"{int(selected_row['反馈人数'])}人")


def render_review_tab(signups_df, results_df, feedbacks_df, thresholds):
    review_df = generate_review_suggestions(
        signups_df, results_df, feedbacks_df,
        thresholds['participation_threshold'],
        thresholds['duration_threshold'],
        thresholds['score_threshold'],
        thresholds['duration_risk_ratio'],
        thresholds['score_risk_ratio'],
        thresholds['total_player_base'],
    )

    st.subheader("📋 活动复盘建议")
    st.info("基于当前侧边栏预警阈值，自动为每场活动生成复盘结论与改进建议，支持按日期和活动名称筛选。")

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        date_range = st.date_input(
            "筛选日期范围",
            value=(review_df['日期'].min(), review_df['日期'].max()),
            min_value=review_df['日期'].min(),
            max_value=review_df['日期'].max(),
            key="review_date_range"
        )
    with col_f2:
        activity_names = st.multiselect(
            "筛选活动名称",
            options=review_df['活动名'].unique(),
            default=review_df['活动名'].unique(),
            key="review_activity_names"
        )

    filtered_df = review_df.copy()
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = filtered_df[
            (filtered_df['日期'] >= start_date) & (filtered_df['日期'] <= end_date)
        ]
    if activity_names:
        filtered_df = filtered_df[filtered_df['活动名'].isin(activity_names)]

    if len(filtered_df) == 0:
        st.warning("没有符合筛选条件的活动")
        return

    high_risk_review = filtered_df[filtered_df['风险等级'] != '低']
    _render_review_high_risk_table(high_risk_review)
    _render_review_detail(filtered_df, thresholds)


def render_raw_data_section(signups_df, results_df, feedbacks_df):
    st.markdown("---")
    st.subheader("📁 原始数据预览")
    data_option = st.selectbox(
        "选择要查看的数据表",
        ["报名记录", "对局结果", "活动反馈"]
    )

    if data_option == "报名记录":
        st.dataframe(signups_df, use_container_width=True)
    elif data_option == "对局结果":
        st.dataframe(
            results_df.drop(columns=['时长状态', '高风险'], errors='ignore'),
            use_container_width=True
        )
    else:
        st.dataframe(
            feedbacks_df.drop(columns=['评分状态', '高风险'], errors='ignore'),
            use_container_width=True
        )


def main():
    configure_page()

    signups_df, results_df, feedbacks_df = load_data()
    thresholds = render_sidebar()

    metrics_df = calculate_activity_metrics(
        signups_df, results_df, feedbacks_df,
        total_player_base=thresholds['total_player_base']
    )

    high_risk_df = get_high_risk_activities(
        signups_df, results_df, feedbacks_df,
        thresholds['participation_threshold'],
        thresholds['duration_threshold'],
        thresholds['score_threshold'],
        thresholds['duration_risk_ratio'],
        thresholds['score_risk_ratio'],
        thresholds['total_player_base'],
    )

    short_games_df = get_short_duration_games(results_df, thresholds['duration_threshold'])
    low_score_df = get_low_score_feedbacks(feedbacks_df, thresholds['score_threshold'])

    render_overview_metrics(metrics_df, high_risk_df, signups_df)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔥 活动热度分析",
        "📈 玩家留存分析",
        "⏱️ 对局时长分布",
        "💬 反馈问题排行",
        "⚠️ 高风险预警",
        "📋 活动复盘建议"
    ])

    with tab1:
        render_popularity_tab(signups_df)
    with tab2:
        render_retention_tab(signups_df)
    with tab3:
        render_duration_tab(results_df, thresholds['duration_threshold'], short_games_df)
    with tab4:
        render_feedback_tab(feedbacks_df, thresholds['score_threshold'], low_score_df)
    with tab5:
        render_risk_tab(metrics_df, high_risk_df, results_df, feedbacks_df, thresholds)
    with tab6:
        render_review_tab(signups_df, results_df, feedbacks_df, thresholds)

    render_raw_data_section(signups_df, results_df, feedbacks_df)


main()
