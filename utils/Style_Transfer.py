# -*- coding: utf-8 -*-
import re
import sys

import pandas as pd
from tqdm import tqdm
import os
from openai import OpenAI


client = OpenAI(
    api_key="sk-f90c7ca300054ccb918e9c54a862f64d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 定义函数查询能力、品质和技能
def style_transfer1(input_text):
    try:
        # 清理和验证输入
        cleaned_JD = clean_and_validate_input(input_text)
        if cleaned_JD == "空文本":
            return "输入为空，请提供有效的简历内容。"

        # 构造提示词
        prompt = f"""作为语言专家，你非常擅长将正式的岗位描述转换成简洁、通俗易懂、更加贴近日常表达的岗位摘要。以下是一个真实岗位描述以及风格转换后的示例，请据此进行类似转换。

        ### 示例：

        岗位描述：

        1. 根据欧美客户的需求和既定的工作流程，在英文系统中准确、高效地进行保险类信息的录入、校对、编写、分析等相关工作，及时出具工作报告；
        2. 通过英文邮件的方式与客户进行及时沟通，建立良好的客户关系。
        任职要求：
        1. 大学英语四级水平，专业不限；
        2. 熟练操作Windows系统，熟悉MS Office软件；
        3. 良好的学习能力；
        4. 良好的团队协作能力及沟通能力；
        5. 工作认真、耐心、责任心强。

        岗位摘要：

        这份工作主要是根据欧美客户的需求，在英文系统中处理保险信息，包括录入、校对、编写和分析，并按时完成报告。同时，你需要通过英文邮件与客户保持沟通，建立良好的合作关系。要求你英语水平达到四级以上（专业不限），熟悉Windows和Office软件，有较强的学习能力和团队合作意识，善于沟通，并且做事认真、细心、有责任感。

        ### 请为以下岗位描述进行风格转换(仅显示摘要内容，无需其他文字)：
        {cleaned_JD}
        """

        # 调用智谱AI的chat功能
        response = client.chat.completions.create(
            model="qwen-max-0125",
            messages=[
                {"role": "user", "content": "作为一个语言专家，你非常擅长用其他风格转述岗位描述的语言。"},
                {"role": "assistant", "content": "当然，请提供岗位描述，我可以为您生成对应文本。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300  # 增加最大token数，以允许更长的输出
        )

        # 检查响应是否有效
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content  # 正确获取内容
        else:
            raise ValueError("API 返回的响应无效或为空。")

    except Exception as e:
        # 打印异常信息并终止程序
        print(f"发生异常: {e}", file=sys.stderr)
        sys.exit(1)  # 终止程序运行

def clean_and_validate_input(input_text):
    """
    对输入内容进行清理和验证，确保其适合传递给大模型。
    """
    # 1. 检查是否为空或仅包含空白字符
    if not input_text.strip():
        return "空文本"

    # 2. 去除多余的空格和特殊字符
    cleaned_text = re.sub(r'\s+', ' ', input_text.strip())  # 合并多余空格
    cleaned_text = re.sub(r'[^\w\s.,;:!?()\-]', '', cleaned_text)  # 移除非标准字符

    # 3. 检测并替换敏感内容（如电话号码、邮箱地址）
    cleaned_text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', cleaned_text)  # 匹配电话号码
    cleaned_text = re.sub(r'[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}', '[REDACTED_EMAIL]', cleaned_text)  # 匹配邮箱地址

    # 4. 截断过长内容以避免超出API限制
    max_length = 500  # 根据需要调整最大长度
    if len(cleaned_text) > max_length:
        cleaned_text = cleaned_text[:max_length] + "..."  # 截断并添加省略号

    return cleaned_text


# def resume_summary_conversion(input_resume):
#     """
#     将简历内容转换为简洁的求职者简介。
#     """
#     # 预处理输入内容
#     cleaned_resume = clean_and_validate_input(input_resume)
#     if cleaned_resume == "空文本":
#         return "输入为空，请提供有效的简历内容。"
#
#     # 构建带有one-shot示例的prompt
#     prompt = f"""作为简历优化专家，你非常擅长将详细简历内容提炼为简明扼要的求职者简介。以下是一个真实的简历信息以及风格转换后的示例，请据此进行类似转换。
#     ### 示例：
#     简历：
#     姓名：宋雨洁 求职岗位：行政专员 年龄：22岁 性别：男 籍贯：上海 工作年限：6年经验
#     电话：15888888888 邮箱：qmjianli@qq.com
#     教育背景：2012-09 ~ 2016-07 全民简历师范大学 工商管理 (本科)，GPA 3.66/4.0（专业前5%）
#     主修课程：基础会计学、货币银行学、统计学、经济法概论、财务会计学、管理学原理、组织行为学、市场营销学、国际贸易理论、人力资源开发与管理、财务管理学、企业经营战略概论、质量管理学等。
#     工作经验：
#     2018-至今 在全民简历科技有限公司担任行政专员，负责行政人事管理、协助总监协调部门工作、执行公司规章制度、督办会议决策事项。
#     2016-2018 在上海斧掌网络科技有限公司担任行政助理，负责财务管理、客户接待、公司班车管理、员工归属感提升、前台管理和招聘工作。
#     技能特长：
#     语言能力：大学英语六级，荣获全国大学生英语竞赛一等奖。
#     计算机：计算机二级证书，熟练操作Windows和MS Office等软件。
#     团队能力：丰富的团队组建与项目管理经验。
#     自我评价：工作积极认真，擅长提出和解决问题，分析能力强，勤奋好学，责任心强，愿迎接挑战。
#
#     简历摘要：
#     宋雨洁，6年行政管理经验的行政专员，熟悉行政和人事管理，协助部门协调及落实公司制度。毕业于全民简历师范大学工商管理专业（GPA 3.66/4.0，专业前5%），具备出色的沟通和团队协作能力。持有大学英语六级证书，计算机二级证书，精通Windows和MS Office办公软件。工作认真负责，有较强的分析和问题解决能力，善于迎接工作中的挑战。
#
#     ### 请为以下简历内容生成简历摘要(仅显示摘要内容，无需其他文字)：
#     {cleaned_resume}
#     """
#
#     try:
#         # 调用智谱AI的chat功能
#         response = client.chat.completions.create(
#             model="qwen-max-0125",
#             messages=[
#                 {"role": "user", "content": "作为一个简历优化专家，你擅长将简历内容转换成简洁的求职者简介。"},
#                 {"role": "assistant", "content": "当然，请提供简历内容，我可以帮助您总结成简介。"},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.7,
#             max_tokens=300  # 增加最大token数，以允许更长的输出
#         )
#         if response and hasattr(response, 'choices') and len(response.choices) > 0:
#             return response.choices[0].message.content  # 正确获取内容
#         return "简历总结转换失败"
#     except Exception as e:
#         # 捕获异常并返回错误信息
#         return f"发生错误：{str(e)}"


# 读取和重命名岗位描述数据
df_job = pd.read_csv("../data/job.csv")

# 设置保存文件路径
output_file = "./data/job_summary.csv"

# 确保保存目录存在
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 如果 CSV 文件已存在，加载已处理的 job_id
if os.path.exists(output_file):
    df_converted = pd.read_csv(output_file)
    processed_job_ids = set(df_converted['job_id'])
else:
    processed_job_ids = set()

# 过滤出尚未处理的数据
df_job_to_process = df_job[~df_job['job_id'].isin(processed_job_ids)].copy()

batch_size = 10

# 分批处理，使用 tqdm 显示进度
for i in tqdm(range(0, len(df_job_to_process), batch_size), desc="Processing batches", unit="batch"):
    batch = df_job_to_process.iloc[i:i + batch_size].copy()
    # 初始化 job_summary 列
    batch['job_summary'] = None

    # 遍历当前批次的每一行
    for idx, row in tqdm(batch.iterrows(), total=len(batch), desc="Processing rows", leave=False):
        job_id = row['job_id']
        # 如果 job_id 已在已处理集合中（可能在并发或其他情况下重复），则跳过
        if job_id in processed_job_ids:
            continue

        try:
            # 调用你自己的函数生成 job_summary
            summary = style_transfer1(row['job_desc'])
            batch.at[idx, 'job_summary'] = summary
            processed_job_ids.add(job_id)
        except Exception as e:
            print(f"处理 job_id {job_id} 时发生错误: {e}", file=sys.stderr)
            continue

    # 筛选出已生成 job_summary 的行，追加保存到 CSV 文件中
    processed_batch = batch[batch['job_summary'].notnull()]
    if not processed_batch.empty:
        processed_batch.to_csv(
            output_file,
            mode='a',
            header=not os.path.exists(output_file),  # 如果文件不存在，则写入表头
            index=False,
            encoding="utf-8"
        )