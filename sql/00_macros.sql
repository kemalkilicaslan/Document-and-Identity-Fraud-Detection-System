-- Turkish national ID (TCKN) check digits:
--   d10 = ((d1+d3+d5+d7+d9)*7 - (d2+d4+d6+d8)) mod 10,  d11 = (d1+..+d10) mod 10
CREATE OR REPLACE MACRO tckn_digit(t, i) AS TRY_CAST(substr(t, i, 1) AS INTEGER);

CREATE OR REPLACE MACRO tckn_valid(t) AS
    t IS NOT NULL
    AND regexp_full_match(t, '[1-9][0-9]{10}')
    AND (((tckn_digit(t,1) + tckn_digit(t,3) + tckn_digit(t,5) + tckn_digit(t,7) + tckn_digit(t,9)) * 7
          - (tckn_digit(t,2) + tckn_digit(t,4) + tckn_digit(t,6) + tckn_digit(t,8))) % 10 + 10) % 10
        = tckn_digit(t,10)
    AND (tckn_digit(t,1) + tckn_digit(t,2) + tckn_digit(t,3) + tckn_digit(t,4) + tckn_digit(t,5)
         + tckn_digit(t,6) + tckn_digit(t,7) + tckn_digit(t,8) + tckn_digit(t,9) + tckn_digit(t,10)) % 10
        = tckn_digit(t,11);


CREATE OR REPLACE MACRO ip24(ip) AS regexp_extract(ip, '^(\d+\.\d+\.\d+)\.', 1);
