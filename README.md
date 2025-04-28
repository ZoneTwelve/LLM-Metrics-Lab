# Guide: LLM Metrics Lab README

Welcome to the **LLM Metrics Lab**, a suite of tools designed to help you evaluate the performance of large language models (LLMs) through standardized metrics and visual analytics. This guide provides comprehensive instructions on how to use the toolkit, interpret the results, and understand the supported metrics.

# **1. Getting Started**

## **1.1 Core Evaluation Toolkit - LLM Metrics Lab**

Before using the LLM Metrics Lab, ensure your environment meets the following requirements:

- Python 3.9+
- Access credentials to the LLM APIs being evaluated

To install the project locally:

```bash
git clone <https://github.com/UnieAI/LLM-Metrics-Lab> llm-metrics-lab
cd llm-metrics-lab
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

# **2. Overview of Available Tools**

## **2.1 Environment setup**

Before running the metrics, make sure to set up your environment variables. Follow the steps below to get started:

**1. Copy the Example Environment File**

First, copy the provided `example.env` to a new `.env` file:

```bash
cp example.env .env
```

**2. Update the `.env` file**

Open the newly created `.env` file and fill in your configuration values:

```
API_URL=https://your-api-url.com
OPENAI_API_KEY=your-api-key
MODEL="gpt-4"
```

|**Variable**|**Description**|
|---|---|
|`API_URL`|The base URL of the API you're using|
|`OPENAI_API_KEY`|Your OpenAI API key|
|`MODEL`|The model's name to be evaluated. (e.g., `gpt-4`, `gpt-3.5-turbo`, `aqua-mini`)|

> 💡 Note: Do not commit your .env file to version control. It contains sensitive information.

Once the `.env` file is configured, you’re ready to run the project 🎉

## **2.2 Metrics Script (**`metrics.py`**)**

To start measuring the performance of a model, run the `metrics.py` script with configurable variables. Below is an example command:

### **2.2.1 Conversation Usage**

```bash
python3 metrics.py \\
  --time-limit=120 \\
  --max-concurrent=2 \\
  --model="SLM-0.5B" \\
  --dataset="unieai/shareGPT" \\
  --conversation="conversation"
```

![image.png](attachment:4ba83172-295d-4577-9859-d304b56f82ce:image.png)

**Dataset Preview:**

You can view the dataset at: [https://huggingface.co/datasets/unieai/shareGPT](https://huggingface.co/datasets/unieai/shareGPT)

|**Column Name**|**Value**|
|---|---|
|random column|random value|
|**conversation**|[|
|{"role": "system", "content": "You are a helpful assistant"},||
|{"role": "user", "content": "Hello world"}||
|]||

### 2.2.2 Template Usage

```bash
python3 metrics.py \\
  --time-limit=120 \\
  --max-concurrent=2 \\
  --model="SLM-0.5B" \\
  --dataset="tatsu-lab/alpaca" \\
  --template="{input}\\nQuestion: {instruction}"
```

**Dataset Preview:**

You can view the dataset at: [https://huggingface.co/datasets/tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

The system transforms the template into a prompt format and feeds it to the LLM. The resulting prompt will be structured as follows:

`[{"role": "user", "content": f"{input}\\nQuestion: {instruction}"}]`

|**Column name**|**Value**|
|---|---|
|id|{index}|
|**input**|Pick a option that most reasonable (1) car (2) ant (3) electron (4) cloud|
|**instruction**|Which one is larger then a regular cup?|
|other_input|{something else}|

### **2.2.3 Variable Explanation:**

|**Variable**|**Description**|
|---|---|
|`--time-limit`|Total duration (in seconds) to run the benchmark. You can adjust this to fit your test window (e.g., `60`, `120`, `300`).|
|`--max-concurrent`|Number of concurrent requests sent to the model. Useful for simulating different load levels. For example:|
|• `--max-concurrent=2`: Low load (ideal for smoke testing).||
|• `--max-concurrent=128`: Medium load (common for typical usage).||
|• `--max-concurrent=512`: High load (used for stress testing or capacity limits).||
|Smaller values for stability checks and increasing gradually to evaluate performance under pressure are recommended.||
|`--model`|The target model to evaluate.|
|`--dataset`|Dataset to use for the evaluation. For example, `unieai/longDocQA`, `unieai/shareGPT`, `tatsu-lab/alpaca`.|
|`--conversation`|Used to extract multi-turn dialogue format from the dataset. Enables simulating realistic chatbot scenarios using dialogue history (e.g., ShareGPT-style messages).|
|_⚠️ Cannot be used together with **--template**._||
|`--template`|Applies a specific template to dataset fields to format the input as a single-turn message, e.g., `[{"role": "user", "content": "template"}]`.|
|Useful for consistent input formatting in tasks like QA, summarization, or instruction following.||
|_⚠️ Cannot be used together with **--conversation**._||

### 2.2.4 Debug Mode Explanation: `CLL` and `FLL`

When running `metrics.py`, you can enable detailed debugging information by setting the following two environment variables. These help developers gain insights into request latency and response behavior. Notice that enabling debug mode may produce a large amount of logging output, so it's recommended to use this during development or troubleshooting.

|**Environment Variable**|**Description**|
|---|---|
|`CLL=debug`|**Console Log Level:** Controls the output of client-side console logs.|
|`FLL=debug`|**File Log Level:** Determines the logging level for entries stored in the `metrics.log` file located in the execution directory.|

**How to Use?**

Set the environment variables before running the command. For example:

```bash
CLL=debug FLL=debug python metrics.py --dataset unieai/shareGPT --conversation conversation --max-concurrent 120 --time-limit 12
```

### **2.2.5 Output:**

After running `metric.py`, the results will be saved as `api_monitor.jsonl` in the execution directory (or in the location specified by `--log-file`). This file contains various performance-related statistics, as illustrated below:

**Terms**

- Session: A session represents a full request-response cycle, from sending a prompt to receiving the final output stream.
- Chunk: A chunk is a partial segment of the streamed response, typically containing several tokens, sent incrementally by the server.

**Example of `api_monitor.jsonl`**

```json
[...,
	{
    "timestamp": "2025-04-21T06:09:41.275064",
    "elapsed_seconds": 9.062501192092896,
    "total_chars": 6548,
    "chars_per_second": 758.1,
    "active_sessions": 5,
    "completed_sessions": 6,
    "total_sessions": 11,
    "successful_requests": 6,
    "failed_requests": 0,
    "tokens_latency": [[], [], [], [0.02373, 0.02211, ... , 0.02632], ...],
    "tokens_amount": [[], [], [], [9, 3, ..., 9], ...],
    "first_token_latencies": [
      1.29894,
      1.23315,
      1.24024,
      1.21885,
      1.20552,
      1.18449,
      1.18725,
      1.20631,
      1.21231,
      1.20483,
      -1
    ]
  },
  ...
]
```

**Fields Explained**

|Field Name|Data Type|Description|
|---|---|---|
|`timestamp`|`str` (ISO format)|The timestamp of when the test ends.|
|`elapsed_seconds`|`float`|The total elapsed time in seconds from the start of the test to the current point.|
|`total_chars`|`int`|The total number of characters received in the current session.|
|`chars_per_second`|`float`|The average characters processed per second.|
|`active_sessions`|`int`|The number of sessions currently running.|
|`completed_sessions`|`int`|The number of sessions that have been completed.|
|`total_sessions`|`int`|The total number of sessions executed (both successful and failed).|
|`successful_requests`|`int`|The number of completed requests.|
|`failed_requests`|`int`|The number of failed requests.|
|`tokens_latency`|`List[List[float]]`|The latency time (in seconds) for each chunk of each session.|
|`tokens_amount`|`List[List[int]]`|The number of tokens in each chunk of each session.|
|`first_token_latencies`|`List[float]`|The latency time for the first chunk of each session.|

- `tokens_latency`: Latency Time for Each Chunk
    - Example:
        
        ```python
        tokens_latency = [
            [0.12, 0.1, 0.15],   # session 0
            [0.09, 0.11, 0.14]   # session 1
        ]
        # Example of Summing Latency for a Single Session (Total latency for all chunks in session 0):
        sum(tokens_latency[0]) = 0.12 + 0.1 + 0.15 = 0.37 seconds
        ```
        
    - `tokens_latency[i][j]` represents the latency (in seconds) between the `j`th chunk and the previous chunk of the `i`th session.
        
    - Each session typically receives multiple chunks, and these values represent the delays between each chunk.
        
    - Smaller latency between chunks indicates a faster response from the streaming service.
        
    - When _tokens_latency[i]_ is empty `tokens_latency[i] = []`, at that point in time, no output was generated.
        
- `tokens_amount`: Number of Tokens in Each Chunk
    - Example:
        
        ```python
        tokens_amount = [
            [5, 7, 3],           # session 0
            [6, 8, 4]            # session 1
        ]
        # Example of Summing Tokens for a Single Session (Total tokens for all chunks in session 1):
        sum(tokens_amount[1]) = 6 + 8 + 4 = 18 tokens
        ```
        
    - `tokens_amount[i][j]` represents the number of tokens (characters) in the `j`th chunk of the `i`th session.
        
    - This value indicates how many tokens are in the content of each chunk, often corresponding to the length of the model's output.
        
    - When _tokens_amount[i]_ is empty `tokens_amount[i] = []` , at that point in time, no output was generated.
        
- `first_token_latencies`Latency for the First Chunk
    - `first_token_latencies[i]` represents the latency time (in seconds) from when the request is sent to when the first chunk of the `i`th session is received.
    - Usually, you only need to observe the latest session’s first token latency to understand the current performance.
    - While `first_tokens_latencies[i]=-1` , at that point in time, no output was generated.

## **2.3 Visualization Script (`visualize.py`)**

You can use `visualize.py` to convert the performance data collected by `metrics.py` (stored in `api_monitor.jsonl`) into a visual image file. This helps you better understand and compare performance metrics across different models or sessions.

### **2.3.1 Usage:**

```bash
python3 visualize.py --log-file <YOUR_LOG_FILE> --output-file <YOUR_OUTPUT_FILE>
```

**Example:**

```bash
python visualize.py --log-file logs/api_monitor_model-SML_concurrent-128.jsonl --output-file visualizations/api_monitor_model-SML_concurrent-128.png
```

The above command visualizes the data from `bash logs/api_monitor_model-SML_concurrent-128.jsonl` and saves the output as `bash visualizations/api_monitor_model-SML_concurrent-128.png`.

### **2.3.2 Variable Explanation:**

|Variable|Description|
|---|---|
|`--log-file`|The input `.jsonl` file generated by `metrics.py` (e.g., `api_monitor.jsonl`).|
|`--output-file`|The name of the output image file (e.g., `api_metrics.png`). It should be a `.png` file.|

# **3. Chart Interpretation**

After running the `visualize.py` script, a chart containing three line graphs will be generated (default file name: `api_metrics.png`).

## **3.1 Chart Introduction**

These three graphs display the performance of the large language model (LLM) over a specific period from different perspectives:

### **3.1.1 Real-Time Output Rate Graph (Characters per Second)**

This graph reflects the model's **output speed** at each moment. A higher value indicates better output efficiency at that time. Fig. 1 shows that the model's output rate fluctuates within a narrow, stable range, indicating consistent generation speed and efficient handling of incoming requests under load.

![Fig. 1: Performance (Characters per Second)](attachment:9cafe578-c1ce-44e2-a2f9-fae1319d9121:image.png)

Fig. 1: Performance (Characters per Second)

### **3.1.2 Total Output Accumulation Graph (Total Characters over Time)**

This graph shows the **cumulative output characters** of the model, meaning the total number of characters the model has output over time. The total number of characters increases steadily over time in Fig. 2, with a consistent slope. This indicates that the model maintains a stable and uniform output rate, reflecting consistent throughput per unit time.

![Fig. 2: Performance (Total Characters over Time)](attachment:8a92e590-4bc2-4358-92b0-54cec77bbace:image.png)

Fig. 2: Performance (Total Characters over Time)

### **3.1.3 Active Sessions Graph (Active Sessions over Time)**

This graph shows how many sessions are active at each moment in time, helping assess how well the system handles concurrent requests over time. The number of active sessions remains consistently high throughout the test period in Fig. 3, which may suggest underutilized resources, as the system does not appear to be reducing session load despite sustained concurrency.

![Fig. 3: Performance (Active Sessions over Time)](attachment:987ec015-56cd-457b-a864-bf6106de24f8:image.png)

Fig. 3: Performance (Active Sessions over Time)

## **3.2 Feature Explanation**

The following sections explain the meaning and interpretation of different features of charts.

### **3.2.1 Interpreting Session Drops in High Concurrency Scenarios**

In Fig. 4, under high concurrency (`max-concurrent = 512`), it shows occasional sharp drops in the Active Sessions chart. However, the Total Characters over Time chart reveals a stable and consistent throughput throughout the measurement window. This indicates that the session drops are not due to instability or resource exhaustion, but are instead a result of the model’s high processing efficiency—requests are completed rapidly, leading to fewer concurrently active sessions.

![Fig. 4: Overall Performance (max-concurrent = 512)](attachment:45fb9c5f-9227-47c5-9442-5cd2c3e16648:image.png)

Fig. 4: Overall Performance (max-concurrent = 512)

### 3.2.2 Throughput Anomaly

Fig. 5 shows a sudden spike in Characters per Second followed by a plateau in Total Characters over Time, suggesting a temporary disruption. This may be attributed to external factors such as shared system usage or network instability, rather than limitations of the model itself.

![Fig. 5: Overall Performance with Anomaly (max-concurrent = 512)](attachment:eac0d706-81d1-46ac-95c9-57b31f7ae0bc:image.png)

Fig. 5: Overall Performance with Anomaly (max-concurrent = 512)

# 4. Troubleshooting

## 4.1 Missing `.env` File Causes `TypeError`

### **Error Message**:

![截圖 2025-04-23 下午5.57.31.png](attachment:bb347157-6d36-4e5f-b908-66831d1656a8:%E6%88%AA%E5%9C%96_2025-04-23_%E4%B8%8B%E5%8D%885.57.31.png)

### **Cause**:

The `.env` file was not properly loaded, resulting in one or more environment variables being `None`, which causes a `TypeError` when concatenated with strings.

### **Solution:**

- Ensure a `.env` file exists in the project root.
- If not, please add it by copy `example.env` and update the variable:
    1. Copy the Example Environment File:
        
        ```bash
        cp example.env .env
        ```
        
    2. Update the `.env` file
        
        ```
        API_URL=https://your-api-url.com
        OPENAI_API_KEY=your-api-key
        MODEL="gpt-4"
        ```
        
        |Variable|Description|
        |---|---|
        |`API_URL`|The base URL of the API you're using|
        |`OPENAI_API_KEY`|Your OpenAI API key|
        |`MODEL`|The model's name to be evaluated.|
        

## 4.2 Missing`--template` or `--conversation` Variable Causes `ValueError`

### **Error Message**:

![截圖 2025-04-23 下午5.55.54.png](attachment:c34c2d90-f645-4234-a386-d764b16670f3:%E6%88%AA%E5%9C%96_2025-04-23_%E4%B8%8B%E5%8D%885.55.54.png)

### **Cause**:

The script was executed without specifying either the `--template` or `--conversation` argument, which are required for the main process to proceed.

### **Solution**:

Provide one of the required arguments when running the command. Please refer to **2.2.1 Conversation Usage** or **2.2.2 Template Usage** section.

## 4.3 Dataset Format Does Not Match Conversation Usage

### Error Message:

![截圖 2025-04-23 下午5.36.02.png](attachment:eef81ec9-906d-473f-8a55-0f801f5687d7:%E6%88%AA%E5%9C%96_2025-04-23_%E4%B8%8B%E5%8D%885.36.02.png)

### **Cause**:

The specified dataset does not conform to the expected format required by the `--conversation` template.

### **Solution**:

Refer to **2.2.1 Conversation Usage / Dataset Preview** section to ensure your dataset structure matches the expected format.

# **Contact and Support**

For issues, questions, or suggestions, please open a GitHub Issue or contact the maintainers at [contact@unieai.com](mailto:contact@unieai.com).

### **Repository**

You can find the full source code, test scripts, and issue tracker at:

- [https://github.com/UnieAI/OpenAI-API-Performance-Metrics](https://github.com/UnieAI/OpenAI-API-Performance-Metrics)