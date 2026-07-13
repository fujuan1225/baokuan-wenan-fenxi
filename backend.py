#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爆款文案分析后端API
支持实时数据更新、预测功能和导出功能
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import json
import datetime
import joblib
import os
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import re
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class ContentAnalysisBackend:
    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.data = None
        self.model = None
        self.vectorizer = None
        self.load_data()
        self.train_model()

    def load_data(self):
        """加载Excel数据"""
        try:
            self.data = pd.read_excel(self.excel_path)
            # 数据清洗
            self.data['播放量'] = pd.to_numeric(self.data['播放量'], errors='coerce')
            self.data['总互动'] = pd.to_numeric(self.data['总互动'], errors='coerce')
            self.data['点赞'] = pd.to_numeric(self.data['点赞'], errors='coerce')
            self.data['评论'] = pd.to_numeric(self.data['评论'], errors='coerce')
            self.data['收藏'] = pd.to_numeric(self.data['收藏'], errors='coerce')

            # 填充缺失值
            numeric_cols = ['播放量', '总互动', '点赞', '评论', '收藏']
            for col in numeric_cols:
                self.data[col] = self.data[col].fillna(0)

            print(f"成功加载数据：{len(self.data)}条记录")
        except Exception as e:
            print(f"数据加载失败：{e}")

    def extract_features(self, text):
        """从文本中提取特征"""
        if pd.isna(text):
            return {}

        text = str(text)
        features = {
            'length': len(text),
            'age_mention': len(re.findall(r'(\d+)[岁过]', text)),
            'disease_count': len(re.findall(r'(腰疼|腿疼|疼痛|突出|麻木|关节)', text)),
            'solution_count': len(re.findall(r'(记好|不用|治疗|方法|方子)', text)),
            'number_count': len(re.findall(r'(\d+)[个种]', text)),
            'risk_words': len(re.findall(r'(特效药|救命药|根治|绝症)', text)),
            'has_exclamation': 1 if '！' in text else 0,
            'has_question': 1 if '?' in text else 0,
        }
        return features

    def train_model(self):
        """训练预测模型"""
        if self.data is None or len(self.data) < 10:
            print("数据不足，无法训练模型")
            return

        # 准备训练数据
        train_data = []
        for idx, row in self.data.iterrows():
            features = self.extract_features(row['发布标题'])
            features['platform_score'] = 1 if row['平台'] == '抖音' else 0.5
            features['status_score'] = 1 if row['进度'] == '已发布' else 0
            train_data.append(features)

        X = pd.DataFrame(train_data)
        y = self.data['总互动']

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 训练模型
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)

        # 评估模型
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"模型训练完成：MAE={mae:.2f}, R²={r2:.3f}")

        # 保存模型
        joblib.dump(self.model, 'content_model.pkl')
        joblib.dump(self.vectorizer, 'vectorizer.pkl')

    def predict_content(self, title, platform='抖音'):
        """预测内容表现"""
        if self.model is None:
            return {'error': '模型未训练'}

        features = self.extract_features(title)
        features['platform_score'] = 1 if platform == '抖音' else 0.5
        features['status_score'] = 1  # 假设都是已发布

        X = pd.DataFrame([features])
        prediction = self.model.predict(X)[0]

        # 计算风险评分
        risk_score = self.calculate_risk_score(title)

        return {
            'predicted_interactions': max(0, prediction),
            'risk_score': risk_score,
            'risk_level': self.get_risk_level(risk_score),
            'optimization_suggestions': self.get_optimization_suggestions(title, risk_score)
        }

    def calculate_risk_score(self, title):
        """计算风险评分"""
        title = str(title)
        score = 0

        # 高风险词
        high_risk_words = ['特效药', '救命药', '根治', '绝症', '包治']
        for word in high_risk_words:
            if word in title:
                score += 30

        # 中风险词
        medium_risk_words = ['不用治疗', '不需要', '不用过度', '秘密', '内幕']
        for word in medium_risk_words:
            if word in title:
                score += 15

        # 过度使用数量词
        numbers = re.findall(r'(\d+)[个种]', title)
        if len(numbers) > 2:
            score += 10

        return min(100, score)

    def get_risk_level(self, score):
        """获取风险等级"""
        if score < 20:
            return 'low'
        elif score < 50:
            return 'medium'
        else:
            return 'high'

    def get_optimization_suggestions(self, title, risk_score):
        """获取优化建议"""
        suggestions = []
        title = str(title)

        # 检查关键词使用
        if '福利药' in title:
            suggestions.append("建议将'福利药'改为'调理方法'或'解决方案'")

        if '特效药' in title:
            suggestions.append("避免使用'特效药'等敏感词，可改为'调理方法'")

        # 检查长度
        if len(title) < 15:
            suggestions.append("标题过短，建议增加到15字以上")
        elif len(title) > 30:
            suggestions.append("标题过长，建议精简到30字以内")

        # 检查年龄表述
        age_count = len(re.findall(r'(\d+)[岁过]', title))
        if age_count > 1:
            suggestions.append("避免连续使用多个年龄表述，可能引起平台算法注意")

        return suggestions if suggestions else ["当前标题已经优化得很好，建议直接发布"]

    def get_viral_templates(self):
        """获取爆款模板"""
        viral_data = self.data.nlargest(10, '总互动')
        templates = []

        for _, row in viral_data.iterrows():
            title = row['发布标题']
            platform = row['平台']
            interactions = row['总互动']

            templates.append({
                'title': title,
                'interactions': interactions,
                'platform': platform,
                'template_type': self.classify_template(title),
                'success_factors': self.extract_success_factors(title)
            })

        return templates

    def classify_template(self, title):
        """分类模板类型"""
        title = str(title)

        if '过了' in title and '记好' in title:
            return '年龄痛点型'
        elif '种' in title and '常见病' in title:
            return '常见病合集型'
        elif '又' in title and '什么' in title:
            return '反常识观点型'
        else:
            return '其他类型'

    def extract_success_factors(self, title):
        """提取成功要素"""
        factors = []
        title = str(title)

        if '过了' in title:
            factors.append('年龄定位精准')
        if '记好' in title:
            factors.append('解决方案明确')
        if '不用' in title:
            factors.append('反常识观点')
        if '种' in title:
            factors.append('信息量大')

        return factors

    def export_report(self, format='json'):
        """导出分析报告"""
        report = {
            '生成时间': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '数据概览': {
                '总内容数': len(self.data),
                '爆款数量': len(self.data[self.data['总互动'] > 10000]),
                '平均播放量': self.data['播放量'].mean(),
                '平均互动量': self.data['总互动'].mean()
            },
            '爆款模板': self.get_viral_templates(),
            '平台分析': self.analyze_platforms(),
            '风险分析': self.analyze_risks()
        }

        if format == 'json':
            return json.dumps(report, ensure_ascii=False, indent=2)
        elif format == 'excel':
            # 导出Excel格式
            with pd.ExcelWriter('analysis_report.xlsx') as writer:
                pd.DataFrame(report['数据概览'], index=[0]).to_excel(writer, sheet_name='数据概览')
                pd.DataFrame(report['爆款模板']).to_excel(writer, sheet_name='爆款模板')
                pd.DataFrame(report['平台分析']).to_excel(writer, sheet_name='平台分析')
                pd.DataFrame(report['风险分析']).to_excel(writer, sheet_name='风险分析')

            return 'analysis_report.xlsx'

    def analyze_platforms(self):
        """分析各平台表现"""
        platform_stats = []

        for platform in self.data['平台'].unique():
            platform_data = self.data[self.data['平台'] == platform]
            stats = {
                '平台': platform,
                '内容数量': len(platform_data),
                '平均播放量': platform_data['播放量'].mean(),
                '平均互动量': platform_data['总互动'].mean(),
                '爆款比例': len(platform_data[platform_data['总互动'] > 10000]) / len(platform_data) * 100
            }
            platform_stats.append(stats)

        return platform_stats

    def analyze_risks(self):
        """分析风险内容"""
        risk_content = self.data[self.data['进度'].isin(['限流', '下架'])]

        risk_analysis = {
            '限流内容总数': len(risk_content),
            '限流率': len(risk_content) / len(self.data) * 100,
            '高风险词统计': self.risk_word_analysis(risk_content['发布标题'].tolist()),
            '常见限流模式': self.find_risk_patterns(risk_content['发布标题'].tolist())
        }

        return risk_analysis

    def risk_word_analysis(self, titles):
        """分析风险词使用频率"""
        word_count = {}
        risk_words = ['特效药', '救命药', '根治', '绝症', '包治', '福利药']

        for title in titles:
            title = str(title)
            for word in risk_words:
                if word in title:
                    word_count[word] = word_count.get(word, 0) + 1

        return word_count

    def find_risk_patterns(self, titles):
        """找出常见风险模式"""
        patterns = {}
        pattern_keywords = [
            ['过了', '记好', '药'],
            ['特效', '治疗', '方法'],
            ['不用', '过度', '治疗']
        ]

        for pattern in pattern_keywords:
            count = 0
            for title in titles:
                title = str(title)
                if all(word in title for word in pattern):
                    count += 1
            patterns[' + '.join(pattern)] = count

        return patterns

# 初始化后端
backend = ContentAnalysisBackend('/Users/xuxu/Desktop/工作簿1.xlsx')

@app.route('/api/data-overview', methods=['GET'])
def get_data_overview():
    """获取数据概览"""
    if backend.data is None:
        return jsonify({'error': '数据未加载'}), 500

    overview = {
        '总内容数': len(backend.data),
        '爆款数量': len(backend.data[backend.data['总互动'] > 10000]),
        '平均播放量': backend.data['播放量'].mean(),
        '平均互动量': backend.data['总互动'].mean(),
        '最高互动量': backend.data['总互动'].max(),
        '更新时间': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return jsonify(overview)

@app.route('/api/predict', methods=['POST'])
def predict_content():
    """预测内容表现"""
    data = request.get_json()
    title = data.get('title', '')
    platform = data.get('platform', '抖音')

    if not title:
        return jsonify({'error': '标题不能为空'}), 400

    result = backend.predict_content(title, platform)
    return jsonify(result)

@app.route('/api/viral-templates', methods=['GET'])
def get_viral_templates():
    """获取爆款模板"""
    templates = backend.get_viral_templates()
    return jsonify(templates)

@app.route('/api/risk-analysis', methods=['GET'])
def get_risk_analysis():
    """获取风险分析"""
    risk_analysis = backend.analyze_risks()
    return jsonify(risk_analysis)

@app.route('/api/export-report', methods=['GET'])
def export_report():
    """导出分析报告"""
    format_type = request.args.get('format', 'json')

    if format_type == 'json':
        report = backend.export_report('json')
        return jsonify(json.loads(report))
    elif format_type == 'excel':
        excel_file = backend.export_report('excel')
        return send_file(
            excel_file,
            as_attachment=True,
            download_name='content_analysis_report.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        return jsonify({'error': '不支持的导出格式'}), 400

@app.route('/api/generate-content', methods=['POST'])
def generate_content():
    """生成内容建议"""
    data = request.get_json()
    audience = data.get('audience', 'female_40')
    disease_type = data.get('disease_type', 'pain')
    style = data.get('style', 'authoritative')

    # 基于参数生成内容建议
    templates = {
        'female_40': {
            'pain': "女性过了40岁，腰疼腿疼肩膀疼，记住这几种调理方法",
            'common': "女性40+常见病清单，中医教你如何预防",
            'chronic': "女性慢性病调理，越早知道越好"
        },
        'male_40': {
            'pain': "男人过了40岁，浑身疼痛不用愁，记住这些方子",
            'common': "中年男性常见健康问题，中医调理指南",
            'chronic': "男性慢性病管理，从中医开始"
        }
    }

    generated_title = templates.get(audience, {}).get(disease_type, "健康养生小知识")

    # 预测表现
    prediction = backend.predict_content(generated_title)

    return jsonify({
        'generated_title': generated_title,
        'prediction': prediction,
        'suggestions': [
            "开头描述痛点，引发共鸣",
            "中间给出3-5个具体解决方案",
            "结尾强调'不需要过度治疗'",
            "加入1-2个真实案例"
        ]
    })

@app.route('/api/real-time-update', methods=['GET'])
def real_time_update():
    """获取实时更新数据"""
    # 模拟实时数据更新
    update_data = {
        '最新内容数': len(backend.data),
        '今日新增': np.random.randint(5, 15),
        '最新爆款': backend.data.nlargest(1, '总互动')['发布标题'].values[0] if len(backend.data) > 0 else "暂无数据",
        '更新时间': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    return jsonify(update_data)

if __name__ == '__main__':
    print("启动爆款文案分析后端服务...")
    print("API文档：")
    print("- GET /api/data-overview - 获取数据概览")
    print("- POST /api/predict - 预测内容表现")
    print("- GET /api/viral-templates - 获取爆款模板")
    print("- GET /api/risk-analysis - 获取风险分析")
    print("- GET /api/export-report - 导出分析报告")
    print("- POST /api/generate-content - 生成内容建议")
    print("- GET /api/real-time-update - 获取实时更新")

    app.run(host='0.0.0.0', port=5000, debug=True)