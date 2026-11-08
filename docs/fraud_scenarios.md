# Fraud Scenarios and Control Matrix

In savings finance the customer first pays a down payment and monthly instalments, and the company finances the home or vehicle at the allocation month. Fraud therefore concentrates at two points:

1. **Application and identity:** a contract signed with a forged identity or someone else's identity.
2. **After allocation (bust-out):** getting allocated quickly and then stopping payment. The loss is the financed amount not covered by collateral.

On the sales side, agents can open fictitious contracts to earn a commission and let them be cancelled.

| Scenario | How it looks in the data | Controls that catch it | Owner |
|---|---|---|---|
| **FORGED_ID**: forged or altered ID document | A rewritten or pasted field on the ID image; sometimes an invalid national ID number; the same identity at several branches; high down payment and fast allocation | R01/R02 document score, R05 ID number, R06 identity velocity, R11 down payment | Operations |
| **SYNTHETIC_ID**: made-up identity | Invalid national ID number, age 19–25, income 2.5–5x the occupation median, random e-mail, online channel, night-time application | R05, R14 young with high income, R15 night online | Operations, Underwriting |
| **FRAUD_RING**: organised fraud | Different identities share a phone, IBAN, address or device; same branch and agent; 4–12 applications within 45 days | R07 phone, R08 IBAN, R09 address, cluster query (`sql/analysis/04`) | Internal Control, Legal |
| **AGENT_FRAUD**: commission fraud | An agent's applications are cancelled 45–100 days later; month-end clustering | R13 agent early cancellation, agent anomaly query (`05`), KRI | Sales, Internal Control |
| **INCOME_FRAUD**: forged income documents | Income 3–6x the occupation median, unexplained high down payment, fast allocation, then default | R10 income outlier, R11 down payment | Underwriting |
| **GUARANTOR_FRAUD**: guarantor abuse | One guarantor on 3–6 unrelated applications; sometimes an invalid guarantor ID number | R12 guarantor reuse | Underwriting, Legal |
| **DEVICE_FARM** (new from 2026-04) | 6–15 identities from one mobile device and one /24 IP network within 10 days. Phone, IBAN and address all differ, so the production rules do not see it | Candidate rules R16 device, R17 IP network; KRI "device seen with 2+ identities" | IT Security, Operations |

## Control design principles

- **Point-in-time:** every feature uses only data available before the application (`sql/01_features.sql`), so the same query runs at decision time.
- **Rules as config:** rules are SQL conditions in `configs/rules.yaml`; changing a threshold needs no code change.
- **Measure before promoting:** new rules go to `configs/rules_candidate.yaml` and `scripts/11_monitoring.py` back-tests them on recent data (recall, precision, extra alert volume).
- **Capacity-based threshold:** the alert threshold follows the analysts' daily review capacity (`scoring.review_capacity_pct`).
- **Deliberate noise:** households at one address, repeat customers, high earners, shared branch tablets, genuine defaults and OCR errors, so no rule is a perfect proxy for the label.
