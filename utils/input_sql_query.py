def get_station_sql_query(station_code, target_date):
    sql_query = f"""            
        WITH base_data AS (
            SELECT 
            station_code,
            ofd_date,
            capped_out_hours as COH,
        
            -- Demand & Planning Deviation
            latest_tva,
            w1_tva,
            daily_updated_cap_target AS duct,
            latest_slammed_volume AS realization,
        
            -- Tactical Capacity Decision
            w3_capacity_ask as w3_caps,
            d1_caps,
            w1_caps,
            caps_change,
            primary_main_reason AS tactical_caps_reason,
        
            -- Operational Execution
            w1_utr,
            w1_otr,
            d1_utr,
            d1_otr,
            d1_mech,
            d1_vs_w1_utr_change,
            d1_otr_vs_w3_cap_ask,
            latest_utilization,
            manual_cap_down,
        
            -- Structural & Systemic Factors
            d1_constraint,
            main_constraint,
            main_constraint_bucket,
            
            -- Contextual Factors
            weather_signal,
            weather_tier,
            backlog_flag,
            total_backlog,
            cf_exclusion_flag,
            co_exclusion_flag,
            prior_3_ofd_weather_flag, 
            ofd_weather_flag,
            EXTRACT(DOW FROM ofd_date) as day_in_week
        FROM "AwsDataCatalog"."ai-assist"."coh_v3_prod"
        WHERE station_code = '{station_code}'
            AND ofd_date = CAST('{target_date}' AS DATE)
            AND country_code = 'US'
            AND cycle = 'CYCLE_1')
        
        SELECT 
            station_code,
            ofd_date,
            COH,
            
            -- Actual vs. Daily Plan Deviation
                (w3_caps - realization) as dev_w3_gap,
                (w1_caps - realization) as dev_w1_gap,
                (d1_caps - realization) as dev_d1_gap,
                (duct - realization) as duct_gap,
            
            -- Features for "Tactical Capacity Decision" Root Causes
                (d1_utr - w1_utr) / w1_utr as utr_capacity_reduction_severity, 
                (d1_otr - w1_otr) / w1_otr as otr_capacity_reduction_severity, 
                CASE WHEN manual_cap_down = 'Y' THEN 1 ELSE 0 END as flag_manual_intervention,

            -- Features for "Operational Execution" Root Causes (Flex & Backlog)
                d1_vs_w1_utr_change,
                d1_otr_vs_w3_cap_ask,
                total_backlog / d1_caps as backlog_pressure,
            
            -- Features for "Structural & Systemic Factors"
                (realization + total_backlog) - d1_caps as total_workload_vs_palanned,
                CASE WHEN cf_exclusion_flag = 'Y' THEN 1 ELSE 0 END as flag_utr_flex,
                CASE WHEN co_exclusion_flag = 'Y' THEN 1 ELSE 0 END as flag_otr_flex,

            -- Features for "Contextual Factors"
                CASE WHEN prior_3_ofd_weather_flag = 'Y' THEN 1 ELSE 0 END as flag_recent_weather_impact,
                CASE WHEN ofd_weather_flag = 'Y' THEN 1 ELSE 0 END as flag_current_date_weather_impact
                -- CASE WHEN days_in_week IN (0, 6) THEN 1 ELSE 0 END as flag_si_weekend
            
            FROM base_data
            ORDER BY station_code, ofd_date;
        """
    return sql_query