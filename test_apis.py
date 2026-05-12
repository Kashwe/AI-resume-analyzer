"""Quick test script to verify all 3 endpoints that were returning 422."""
import io
import requests
from fpdf import FPDF

BASE = "http://127.0.0.1:8000"

# --- Create a sample resume PDF in-memory ---
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=12)
lines = [
    "John Doe",
    "john.doe@email.com | +1-555-123-4567 | linkedin.com/in/johndoe",
    "",
    "WORK EXPERIENCE",
    "",
    "Senior Software Engineer at Acme Corp (2020 - Present)",
    "- Developed microservices using Python, FastAPI, and Docker",
    "- Reduced deployment time by 40 percent",
    "- Led a team of 5 engineers to deliver a real-time analytics platform",
    "- Architected event-driven systems with Kafka and PostgreSQL",
    "",
    "Software Engineer at Beta Inc (2017 - 2020)",
    "- Built REST APIs serving 10M+ requests/day using Django and Redis",
    "- Implemented CI/CD pipelines with GitHub Actions",
    "- Cutting release cycles by 50 percent",
    "- Mentored 3 junior developers on best practices",
    "",
    "EDUCATION",
    "B.S. Computer Science - State University (2013 - 2017)",
    "",
    "SKILLS",
    "Python, JavaScript, TypeScript, React, FastAPI, Django, Docker,",
    "Kubernetes, PostgreSQL, Redis, Kafka, AWS, Git, CI/CD, Agile",
    "",
    "CERTIFICATIONS",
    "AWS Certified Solutions Architect - Associate",
]
for line in lines:
    pdf.cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
pdf_bytes = pdf.output()

# --- Test 1: /extract-skills ---
print("=" * 60)
print("TEST 1: /extract-skills")
print("=" * 60)
resp = requests.post(
    f"{BASE}/extract-skills",
    files={"file": ("resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    timeout=60,
)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Technical skills: {data.get('technical_skills', [])[:5]}...")
    print(f"Experience level: {data.get('experience_level')}")
else:
    print(f"ERROR: {resp.text[:500]}")

# --- Test 2: /ats-score ---
print("\n" + "=" * 60)
print("TEST 2: /ats-score")
print("=" * 60)
resp = requests.post(
    f"{BASE}/ats-score",
    files={"file": ("resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    timeout=60,
)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Overall score: {data.get('overall_score')}/100")
    print(f"Verdict: {data.get('verdict')}")
else:
    print(f"ERROR: {resp.text[:500]}")

# --- Test 3: /match-job ---
print("\n" + "=" * 60)
print("TEST 3: /match-job")
print("=" * 60)
job_desc = (
    "Senior Backend Engineer - TechCorp. "
    "We are looking for a Senior Backend Engineer to join our platform team. "
    "Requirements: 5+ years of experience with Python and backend frameworks "
    "(Django, FastAPI). Strong experience with PostgreSQL and Redis. "
    "Experience with Docker and Kubernetes. Familiarity with CI/CD pipelines "
    "and cloud services (AWS preferred). Strong problem-solving and communication skills."
)
resp = requests.post(
    f"{BASE}/match-job",
    files={"file": ("resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    data={"job_description": job_desc},
    timeout=60,
)
print(f"Status: {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"Match score: {data.get('match_score')}/100")
    print(f"Matched: {data.get('matched_skills', [])[:5]}")
    print(f"Missing: {data.get('missing_skills', [])}")
else:
    print(f"ERROR: {resp.text[:500]}")

print("\n" + "=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
