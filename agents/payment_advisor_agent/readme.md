payment_analytics_coordinator (ROOT AGENT - LlmAgent)
│
├── main_kpi_agent (BaseAgent + BigQueryTool)
│   ├── Purpose: Calculate core payment KPIs
│   ├── Tables: purchase_transaction, chargebacks, token_redeem_grade
│   └── Output: Success rates, volumes, chargeback ratios, redeem distributions
│
├── deep_dive_agent (BaseAgent + BigQueryTool)  
│   ├── Purpose: Investigate payment issues and root causes
│   ├── Tables: chargebacks, purchase_transaction, user_profile, main_kpi_dataset
│   ├── Triggers: "CHARGEBACK INVESTIGATION NEEDED" | "PERFORMANCE INVESTIGATION NEEDED"
│   └── Output: Root cause analysis, pattern identification, recommendations
│
└── visualization_manager_agent (BaseAgent)
    ├── Purpose: Create CSV exports, charts, and executive summaries
    ├── Data Source: Session state from KPI and Deep Dive analysis
    ├── Tools: No external tools (works with in-memory data)
    └── Output: CSV files, visualizations, business reports