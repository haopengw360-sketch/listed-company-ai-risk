CREATE INDEX IF NOT EXISTS idx_financial_code_year
ON financial_panel(code, year);

CREATE INDEX IF NOT EXISTS idx_text_code_year
ON text_metrics(code, year);

DROP VIEW IF EXISTS financial_text_features;

CREATE VIEW financial_text_features AS
SELECT
    f.code,
    f.year,
    f.size,
    f.lev,
    f.roa,
    f.growth,
    f.cashflow,
    f.tangibility,
    f.bm,
    f.turnover,
    f.ret,
    f.sigma,
    t.char_count,
    t.risk_freq,
    t.negative_freq,
    t.positive_freq,
    t.uncertainty_freq,
    t.vague_freq,
    t.risk_tone
FROM financial_panel AS f
INNER JOIN text_metrics AS t
    ON f.code = t.code AND f.year = t.year;
