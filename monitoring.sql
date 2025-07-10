-- File: create_monitoring_db.sql (run this alone)
CREATE DATABASE monitoring_db OWNER airflow;

-- File: grant_permissions.sql
-- Connect to the new database before running this
GRANT ALL PRIVILEGES ON DATABASE monitoring_db TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO airflow;