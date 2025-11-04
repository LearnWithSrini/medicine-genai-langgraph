CREATE TABLE IF NOT EXISTS bi.dim_drug (
    drug_id             STRING,
    brand_name          STRING,
    inn_name            STRING,
    atc_code            STRING,
    therapeutic_area    STRING,
    route               STRING,
    dosage_form         STRING,
    first_approval_date DATE,
    is_marketed         BOOLEAN
)
USING delta;

CREATE TABLE IF NOT EXISTS bi.dim_disease (
    disease_id        STRING,
    name              STRING,
    icd10_code        STRING,
    therapeutic_area  STRING
)
USING delta;

CREATE TABLE IF NOT EXISTS bi.bridge_drug_disease (
    drug_id        STRING,
    disease_id     STRING,
    line_of_therapy STRING,
    region         STRING,
    status         STRING
)
USING delta;

CREATE TABLE IF NOT EXISTS bi.fact_trial_outcome (
    trial_id       STRING,
    drug_id        STRING,
    disease_id     STRING,
    endpoint_code  STRING,
    hr             DOUBLE,
    ci_low         DOUBLE,
    ci_high        DOUBLE,
    p_value        DOUBLE,
    n_patients     INT,
    is_primary     BOOLEAN
)
USING delta;
