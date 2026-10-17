# Revised AWS Deployment Architecture & Sizing Guide

**Project:** InterviewIQ — AI Interview Coach  
**Target Environment:** AWS Asia Pacific (Sydney) — `ap-southeast-2`  
**Execution Type:** ECS on AWS Fargate (Express / Serverless Container Architecture)

---

## 1. Architecture Overview

```
                        ┌──────────────────────────────────────────────┐
                        │              Browser / Candidate             │
                        └──────────────────────┬───────────────────────┘
                                               │
                        ┌──────────────────────▼───────────────────────┐
                        │         Amazon CloudFront CDN (HTTPS)        │
                        └──────────────┬───────────────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │ Static SPA                                  │ /api/* Requests
                ▼                                             ▼
┌───────────────────────────────┐             ┌────────────────────────────────┐
│       Amazon S3 Bucket        │             │   Application Load Balancer    │
│    (Compiled React Build)     │             │             (ALB)              │
└───────────────────────────────┘             └───────────────┬────────────────┘
                                                              │
                                              ┌───────────────▼────────────────┐
                                              │      Amazon ECS on Fargate     │
                                              │  (2 vCPU, 8 GB RAM Container)  │
                                              │   FastAPI + Strands Decider    │
                                              └───────────────┬────────────────┘
                                                              │
                                                              │ IAM Task Role (Converse)
                                                              ▼
                                              ┌────────────────────────────────┐
                                              │         Amazon Bedrock         │
                                              │      (google.gemma-3-4b-it)    │
                                              └────────────────────────────────┘
```

---

## 2. Model Memory & Sizing Analysis

### Strands Decider 2B Checkpoint Composition
* **Base Torso:** `Qwen/Qwen3.5-2B-Base` (torso weights: **~4.2 GB** unquantized safetensors).
* **LoRA Adapter:** `StrandsAgents/strands-decider-2B-hobson-v21/lora` (`adapter_model.safetensors`).
* **Scoring Head:** `head.safetensors` (Pointer head with 24 slot embeddings + temperature calibration).

### Fargate Sizing Determination
* **In-Memory Footprint:** Loading torso + adapter + pointer head in PyTorch float32/bfloat16 allocates **~3.7 GB to 4.2 GB** memory.
* **Peak Activation Memory:** Forward pass evaluation requires ~500 MB – 1 GB scratch space.
* **Allocated Sizing:** **2 vCPU, 8 GB Memory** (Standard Fargate compute profile).
  * *Why 8 GB?* A 4 GB allocation would trigger Out-Of-Memory (OOM) killing during initial model weight loading or multi-request processing. 8 GB provides a stable 3.8 GB safety headroom.

---

## 3. Comprehensive Cost Estimates (ap-southeast-2 Sydney)

Prices derived from the official AWS Pricing Calculator for `ap-southeast-2`:
* **AWS Fargate (Linux x86):**
  * vCPU: \$0.04656 per vCPU-hour
  * Memory: \$0.00511 per GB-hour
  * 2 vCPU + 8 GB RAM per hour = `(2 * $0.04656) + (8 * $0.00511)` = **\$0.1340 per hour**
* **Application Load Balancer (ALB):**
  * \$0.0252 per hour + \$0.008 per LCU-hour ≈ **\$0.026 per hour**
* **Amazon ECR Storage:**
  * \$0.10 per GB/month (for ~5 GB model container ≈ \$0.50/month)
* **CloudWatch Logs:**
  * \$0.50 per GB ingested
* **Amazon S3 + CloudFront:**
  * Free Tier (1 TB data transfer + 5 GB storage) ≈ **\$0.00**
* **Amazon Bedrock (`google.gemma-3-4b-it`):**
  * ~$0.00015 per 1,000 input tokens, ~$0.0006 per 1,000 output tokens.
  * For 200 interview sessions: **~$0.15**

### Projected Cost Breakdown

| Duration | Fargate Compute (2 vCPU, 8 GB) | ALB | ECR / Logs / Transfer | Bedrock Token Estimate | Total Expected Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **24 Hours** (1 day demo) | \$3.22 | \$0.62 | \$0.20 | \$0.15 | **\$4.19** |
| **3 Days** (72 hours active) | \$9.65 | \$1.87 | \$0.35 | \$0.30 | **\$12.17** |
| **7 Days** (Full hackathon review) | \$22.51 | \$4.37 | \$0.65 | \$0.50 | **\$28.03** |

*Cost mitigation note: Fargate tasks and the ALB can be stopped or scaled to 0 tasks via the AWS Console or CLI when not testing to stop all compute charges.*

---

## 4. Minimum IAM Permissions (Task Execution & Runtime)

### A. ECS Task Role (Container Runtime Permissions)
Attached to the running container to allow calling Bedrock without long-lived keys in `.env`:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:Converse"
      ],
      "Resource": "arn:aws:bedrock:ap-southeast-2::foundation-model/google.gemma-3-4b-it"
    }
  ]
}
```

### B. ECS Task Execution Role (Startup & Logging)
Pre-managed AWS managed policy: `arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy` (Allows pulling image from ECR and streaming logs to CloudWatch).
