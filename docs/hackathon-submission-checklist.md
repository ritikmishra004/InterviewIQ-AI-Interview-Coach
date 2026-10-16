# Hackathon Submission Checklist

**Project:** InterviewIQ — Adaptive AI Interview Agent  
**Event:** AWS Strands Decider Agent Hackathon (October 2026)  
**Required Tags:** `#strands-decider-2b` `#policy`

---

## 1. Compliance Checklist

| Item | Requirement | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Model** | Uses Strands Decider 2B | ✅ Completed | Official `strands-decider` ChoiceQuestion API in `backend/app/services/decider_service.py`. |
| **Bedrock** | Paired with Amazon Bedrock | ✅ Completed | Boto3 Converse API targeting `google.gemma-3-4b-it` in `ap-southeast-2` (`bedrock_service.py`). |
| **Pattern** | Reusable Hackathon Pattern | ✅ Completed | `#policy` pattern in `patterns/adaptive_interview_policy/`. |
| **License** | Approved OSI Open Source | ✅ Completed | Apache-2.0 License added (`LICENSE`). |
| **Tags** | Submission Tags | ✅ Completed | `#strands-decider-2b` and `#policy` across code, docs, and UI. |
| **Config** | Safe Secret Isolation | ✅ Completed | Environment template `.env.example` provided. |
| **Testing** | Automated Test Suite | ✅ Completed | 25 automated tests passing in Pytest (`backend/tests/`). |
| **Frontend** | Build Integrity | ✅ Completed | Vite React production build passing without errors. |
| **Article** | AWS Builder Center Draft | ✅ Completed | Formatted draft in `docs/builder-center-article-draft.md`. |
| **Video** | Demo Video Script (<3 min) | ✅ Completed | Time-coded script in `docs/demo-video-script.md`. |
| **Submission Window** | Target Hackathon Dates | ℹ️ Pending Submission | Target window: October 16–23, 2026 (Verify official portal rules). |

---

## 2. Lowest-Cost AWS Deployment Architecture
- **Backend API**: AWS App Runner container or ECS Fargate (1 vCPU, 2 GB RAM). Incurs $0 when paused.
- **Frontend**: AWS S3 Bucket + CloudFront distribution (Static hosting within AWS Free Tier).
- **Bedrock**: Pay-as-you-go per token with zero idle cost.
