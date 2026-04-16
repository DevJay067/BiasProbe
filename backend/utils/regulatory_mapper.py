"""
regulatory_mapper.py
====================
BiasProbe — Maps bias findings to specific regulatory citations.

Each significant statistical finding (attribute × scenario × dimension)
is matched against a lookup table of regulatory frameworks.  The mapper
returns a deduplicated list of RegulatoryFlag objects that can be embedded
in the StatisticalReport and shown in the audit PDF.

Supported frameworks
--------------------
  EU AI Act (2024)             — Article 5 (prohibited), Article 10 (data governance)
  EEOC / Title VII (US)        — Employment discrimination
  US Fair Housing Act          — Housing/lending discrimination by race, national origin
  Equal Credit Opportunity Act — Lending discrimination by any protected class
  Age Discrimination in
    Employment Act (ADEA)      — Age discrimination in hiring (40+)
  Age Discrimination Act (US)  — Age discrimination in health services
  UK Equality Act 2010         — Nine protected characteristics
  GDPR Article 22              — Automated decision-making affecting individuals
  US Civil Rights Act          — Broad anti-discrimination basis
  HIPAA / ACA §1557            — Health non-discrimination
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.stats_engine import DimensionResult


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RegulatoryFlag:
    """A single regulatory citation triggered by a bias finding."""
    regulation_id:   str    # short key, e.g. "EU_AI_ACT_ART10"
    regulation_name: str    # full human-readable name
    article:         str    # specific article / section
    description:     str    # one-sentence summary of what this covers
    severity:        str    # "prohibited" | "high" | "medium" | "informational"
    url:             str    # reference link

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Regulation definitions (canonical registry)
# ---------------------------------------------------------------------------

_REGS: dict[str, RegulatoryFlag] = {
    "EU_AI_ACT_ART5": RegulatoryFlag(
        regulation_id="EU_AI_ACT_ART5",
        regulation_name="EU AI Act",
        article="Article 5 — Prohibited AI Practices",
        description=(
            "Prohibits AI systems that exploit characteristics of "
            "vulnerable groups or use subliminal manipulation."
        ),
        severity="prohibited",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
    ),
    "EU_AI_ACT_ART10": RegulatoryFlag(
        regulation_id="EU_AI_ACT_ART10",
        regulation_name="EU AI Act",
        article="Article 10 — Data and Data Governance",
        description=(
            "High-risk AI systems must use training data that is free from "
            "bias with respect to protected characteristics."
        ),
        severity="high",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
    ),
    "EU_AI_ACT_ART13": RegulatoryFlag(
        regulation_id="EU_AI_ACT_ART13",
        regulation_name="EU AI Act",
        article="Article 13 — Transparency and Provision of Information",
        description=(
            "High-risk AI systems must be transparent enough to allow "
            "human oversight of discriminatory outcomes."
        ),
        severity="high",
        url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
    ),
    "EEOC_TITLE_VII": RegulatoryFlag(
        regulation_id="EEOC_TITLE_VII",
        regulation_name="EEOC / Title VII of the Civil Rights Act (US)",
        article="42 U.S.C. § 2000e",
        description=(
            "Prohibits employment discrimination based on race, color, "
            "religion, sex, or national origin."
        ),
        severity="high",
        url="https://www.eeoc.gov/statutes/title-vii-civil-rights-act-1964",
    ),
    "ADEA": RegulatoryFlag(
        regulation_id="ADEA",
        regulation_name="Age Discrimination in Employment Act (US)",
        article="29 U.S.C. § 623",
        description=(
            "Prohibits employment discrimination against individuals aged 40 or older."
        ),
        severity="high",
        url="https://www.eeoc.gov/statutes/age-discrimination-employment-act-1967",
    ),
    "FAIR_HOUSING_ACT": RegulatoryFlag(
        regulation_id="FAIR_HOUSING_ACT",
        regulation_name="US Fair Housing Act",
        article="42 U.S.C. § 3604",
        description=(
            "Prohibits discrimination in the sale, rental, or financing of housing "
            "based on race, color, national origin, religion, sex, familial status, or disability."
        ),
        severity="high",
        url="https://www.hud.gov/program_offices/fair_housing_equal_opp/fair_housing_act_overview",
    ),
    "ECOA": RegulatoryFlag(
        regulation_id="ECOA",
        regulation_name="Equal Credit Opportunity Act (US)",
        article="15 U.S.C. § 1691",
        description=(
            "Prohibits creditors from discriminating on the basis of race, color, religion, "
            "national origin, sex, marital status, age, or receipt of public assistance."
        ),
        severity="high",
        url="https://www.consumerfinance.gov/consumer-tools/educator-tools/youth-financial-education/teach/activities/equal-credit-opportunity-act/",
    ),
    "UK_EQUALITY_ACT": RegulatoryFlag(
        regulation_id="UK_EQUALITY_ACT",
        regulation_name="UK Equality Act 2010",
        article="Section 4 — Protected Characteristics",
        description=(
            "Prohibits discrimination on the basis of nine protected characteristics "
            "including age, race, sex, religion, and disability."
        ),
        severity="high",
        url="https://www.legislation.gov.uk/ukpga/2010/15/contents",
    ),
    "GDPR_ART22": RegulatoryFlag(
        regulation_id="GDPR_ART22",
        regulation_name="GDPR",
        article="Article 22 — Automated Individual Decision-Making",
        description=(
            "Individuals have the right not to be subject to decisions based solely "
            "on automated processing that significantly affects them."
        ),
        severity="high",
        url="https://gdpr-info.eu/art-22-gdpr/",
    ),
    "ACA_1557": RegulatoryFlag(
        regulation_id="ACA_1557",
        regulation_name="Affordable Care Act — Section 1557",
        article="42 U.S.C. § 18116",
        description=(
            "Prohibits discrimination on the basis of race, color, national origin, "
            "sex, age, or disability in health programs or activities receiving federal funding."
        ),
        severity="high",
        url="https://www.hhs.gov/civil-rights/for-individuals/section-1557/index.html",
    ),
    "AGE_DISCRIMINATION_ACT": RegulatoryFlag(
        regulation_id="AGE_DISCRIMINATION_ACT",
        regulation_name="Age Discrimination Act of 1975 (US)",
        article="42 U.S.C. § 6102",
        description=(
            "Prohibits discrimination based on age in programs or activities "
            "receiving federal financial assistance."
        ),
        severity="medium",
        url="https://www.hhs.gov/sites/default/files/ocr/civilrights/resources/factsheets/age.pdf",
    ),
    "CFPB_UDAP": RegulatoryFlag(
        regulation_id="CFPB_UDAP",
        regulation_name="CFPB — Unfair, Deceptive, or Abusive Acts or Practices",
        article="12 U.S.C. § 5531",
        description=(
            "Prohibits unfair, deceptive, or abusive practices by financial service providers, "
            "which may include algorithmically differential treatment."
        ),
        severity="medium",
        url="https://www.consumerfinance.gov/compliance/supervisory-guidance/unfair-deceptive-abusive-acts-practices/",
    ),
    "FTC_ACT_5": RegulatoryFlag(
        regulation_id="FTC_ACT_5",
        regulation_name="FTC Act — Section 5",
        article="15 U.S.C. § 45",
        description=(
            "The FTC has indicated that AI bias resulting in differential consumer "
            "treatment may constitute an unfair or deceptive trade practice."
        ),
        severity="informational",
        url="https://www.ftc.gov/business-guidance/blog/2021/04/aiming-truth-fairness-equity-ftcs-approach-commercial-surveillance-data-security",
    ),
}


# ---------------------------------------------------------------------------
# Lookup table: (scenario_family, attribute) → list of regulation_ids
# ---------------------------------------------------------------------------

# Scenario families normalise the 5 template scenarios into broader domains
_SCENARIO_FAMILY: dict[str, str] = {
    "hiring_assistant":  "employment",
    "loan_advisor":      "lending",
    "medical_triage":    "healthcare",
    "customer_support":  "consumer",
    "content_moderator": "content",
}

# (scenario_family, attribute) → regulation ids
# Entries with attribute="*" match any attribute in that scenario family.
_LOOKUP: list[tuple[str, str, list[str]]] = [
    # ---- Employment ----
    ("employment", "gender",   ["EU_AI_ACT_ART10", "EU_AI_ACT_ART13", "EEOC_TITLE_VII", "UK_EQUALITY_ACT", "GDPR_ART22"]),
    ("employment", "race",     ["EU_AI_ACT_ART10", "EU_AI_ACT_ART13", "EEOC_TITLE_VII", "UK_EQUALITY_ACT", "GDPR_ART22"]),
    ("employment", "age",      ["EU_AI_ACT_ART10", "EU_AI_ACT_ART13", "ADEA",           "UK_EQUALITY_ACT", "GDPR_ART22"]),
    ("employment", "religion", ["EU_AI_ACT_ART10", "EEOC_TITLE_VII",  "UK_EQUALITY_ACT"]),
    ("employment", "*",        ["EU_AI_ACT_ART13", "GDPR_ART22"]),

    # ---- Lending ----
    ("lending", "race",              ["EU_AI_ACT_ART10", "FAIR_HOUSING_ACT", "ECOA", "CFPB_UDAP", "FTC_ACT_5"]),
    ("lending", "gender",            ["EU_AI_ACT_ART10", "ECOA",             "UK_EQUALITY_ACT"]),
    ("lending", "age",               ["EU_AI_ACT_ART10", "ECOA",             "UK_EQUALITY_ACT", "CFPB_UDAP"]),
    ("lending", "marital_status",    ["ECOA"]),
    ("lending", "national_origin",   ["FAIR_HOUSING_ACT", "ECOA"]),
    ("lending", "*",                 ["EU_AI_ACT_ART13", "GDPR_ART22"]),

    # ---- Healthcare ----
    ("healthcare", "race",             ["EU_AI_ACT_ART10", "ACA_1557",             "UK_EQUALITY_ACT"]),
    ("healthcare", "gender",           ["EU_AI_ACT_ART10", "ACA_1557",             "UK_EQUALITY_ACT"]),
    ("healthcare", "age",              ["EU_AI_ACT_ART10", "ACA_1557", "AGE_DISCRIMINATION_ACT", "UK_EQUALITY_ACT"]),
    ("healthcare", "insurance_status", ["ACA_1557"]),
    ("healthcare", "language",         ["ACA_1557"]),
    ("healthcare", "*",                ["EU_AI_ACT_ART5", "EU_AI_ACT_ART13", "GDPR_ART22"]),

    # ---- Consumer services ----
    ("consumer", "race",          ["EU_AI_ACT_ART10", "FTC_ACT_5", "UK_EQUALITY_ACT"]),
    ("consumer", "gender",        ["EU_AI_ACT_ART10", "FTC_ACT_5", "UK_EQUALITY_ACT"]),
    ("consumer", "age",           ["EU_AI_ACT_ART10", "FTC_ACT_5", "UK_EQUALITY_ACT"]),
    ("consumer", "writing_style", ["EU_AI_ACT_ART10", "FTC_ACT_5"]),
    ("consumer", "*",             ["EU_AI_ACT_ART13", "GDPR_ART22"]),

    # ---- Content moderation ----
    ("content", "religion",              ["EU_AI_ACT_ART5", "EU_AI_ACT_ART10"]),
    ("content", "race",                  ["EU_AI_ACT_ART5", "EU_AI_ACT_ART10"]),
    ("content", "gender",                ["EU_AI_ACT_ART5", "EU_AI_ACT_ART10"]),
    ("content", "political_affiliation", ["EU_AI_ACT_ART5", "EU_AI_ACT_ART10"]),
    ("content", "nationality",           ["EU_AI_ACT_ART5", "EU_AI_ACT_ART10"]),
    ("content", "*",                     ["EU_AI_ACT_ART13"]),
]


# ---------------------------------------------------------------------------
# RegulatoryMapper
# ---------------------------------------------------------------------------

class RegulatoryMapper:
    """
    Maps bias findings to regulatory citations.

    Usage
    -----
    >>> mapper = RegulatoryMapper()
    >>> flags = mapper.map(scenario="hiring_assistant", attribute="gender", findings=[...])
    """

    def map(
        self,
        scenario: str,
        attribute: str,
        findings: list["DimensionResult"],
    ) -> list[RegulatoryFlag]:
        """
        Return deduplicated RegulatoryFlag objects for a set of significant findings.

        Only returns flags if at least one finding is statistically significant.
        If there are no significant findings, returns an empty list.

        Parameters
        ----------
        scenario  : str   — e.g. "hiring_assistant"
        attribute : str   — e.g. "gender"
        findings  : list  — significant DimensionResult objects

        Returns
        -------
        list[RegulatoryFlag]  — deduplicated, sorted by severity
        """
        if not findings:
            return []

        family = _SCENARIO_FAMILY.get(scenario, "general")
        reg_ids: set[str] = set()

        # Collect IDs from attribute-specific and wildcard rules
        for (fam, attr, ids) in _LOOKUP:
            if fam != family:
                continue
            if attr == attribute or attr == "*":
                reg_ids.update(ids)

        # For highly significant large-effect findings, also add EU AI Act Art 5
        # (potential prohibited practice if scope is narrow)
        has_large_high = any(
            f.effect_size_label == "large" and f.is_highly_significant
            for f in findings
        )
        if has_large_high and family in ("healthcare", "content"):
            reg_ids.add("EU_AI_ACT_ART5")

        # Build flag objects, deduplicated and severity-sorted
        severity_order = {"prohibited": 0, "high": 1, "medium": 2, "informational": 3}
        flags = sorted(
            [_REGS[rid] for rid in reg_ids if rid in _REGS],
            key=lambda r: severity_order.get(r.severity, 9),
        )
        return flags

    @staticmethod
    def all_regulations() -> list[RegulatoryFlag]:
        """Return the full regulatory registry (useful for UI display)."""
        return list(_REGS.values())

    @staticmethod
    def get_regulation(regulation_id: str) -> RegulatoryFlag | None:
        """Look up one regulation by its ID."""
        return _REGS.get(regulation_id)
