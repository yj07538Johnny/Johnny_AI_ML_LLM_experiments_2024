# Container Management System - Implementation Checklist

**Project:** Multi-user container deployment portal with admin dashboard and monitoring
**Location:** `/home/jlmorg1/work/Johnny_AI_ML_LLM_experiments_2024/container-management/`
**Started:** 2026-01-27

---

## What We're Building (The Simple Version)

```
Users go to web page → Pick a container → Click deploy → Container runs on VM
Admin sees dashboard → Who has what → Stop idle stuff → Monitor everything
Prometheus watches VMs → Grafana shows pretty graphs → SA sees problems
```

---

## THE CHECKLIST (Check boxes when done)

### Phase 1: Make the Database Work

```
Directory: streamlit-service/
```

- [ ] **1.1 Create models.py**
  ```
  File: streamlit-service/models.py
  What: Data classes for User, Deployment, DeploymentStatus, HealthCheck
  Test: python3 -c "from models import User, Deployment; print('models OK')"
  ```

- [ ] **1.2 Create database.py**
  ```
  File: streamlit-service/database.py
  What: SQLite database with users table, deployments table, health_checks table
  Test: python3 -c "from database import get_db; db=get_db(); print('database OK')"
  ```

- [ ] **1.3 Test database independently**
  ```
  Run: cd streamlit-service && python3 -c "
  from database import get_db
  from models import User
  db = get_db()
  user = User(username='testuser', email='test@test.com')
  user = db.get_or_create_user(user)
  print(f'Created user: {user.username} with id {user.id}')
  "
  ```

---

### Phase 2: Update the Streamlit App

```
Directory: streamlit-service/
```

- [ ] **2.1 Update .env file**
  ```
  File: streamlit-service/.env
  Add these lines:
    ENABLE_SIMPLE_AUTH=true
    ADMIN_USERS=jlmorg1,admin
    DB_PATH=/home/jlmorg1/.container-deployments.db
  ```

- [ ] **2.2 Update requirements.txt**
  ```
  File: streamlit-service/requirements.txt
  Add: pandas>=2.0.0
  Run: pip install pandas
  ```

- [ ] **2.3 Add simple auth to app.py**
  ```
  File: streamlit-service/app.py
  What: Add get_or_select_user() function for username dropdown/entry
  Where: After the existing get_user_from_pki() function
  ```

- [ ] **2.4 Add database integration to app.py**
  ```
  File: streamlit-service/app.py
  What: Import database, save deployments to SQLite after successful deploy
  Where: In the deploy button handler, after result["success"]
  ```

- [ ] **2.5 Add Admin Dashboard tab to app.py**
  ```
  File: streamlit-service/app.py
  What: New tab with metrics, deployment list, user table, health checks
  Where: In main() function, add tab4 for admins
  ```

- [x] **2.6 Test the updated app**
  ```
  Run: cd streamlit-service && streamlit run app.py --server.port 8501
  Tested 2026-01-28:
    ✓ App running on port 8501
    ✓ Health endpoint responding
    ✓ Models and database imports working
    ✓ Environment variables loaded (ENABLE_SIMPLE_AUTH, ADMIN_USERS)
    ✓ SQLite database operational (2 users, 1 deployment tracked)
  ```

---

### Phase 3: Create Monitoring Stack ✅ COMPLETE

```
Directory: /home/jlmorg1/monitoring/
```

- [x] **3.1 Create monitoring directory structure**
  ```
  Run:
  mkdir -p /home/jlmorg1/monitoring/prometheus
  mkdir -p /home/jlmorg1/monitoring/grafana/provisioning/dashboards
  mkdir -p /home/jlmorg1/monitoring/grafana/provisioning/datasources
  mkdir -p /home/jlmorg1/monitoring/grafana/dashboards
  mkdir -p /home/jlmorg1/monitoring/alertmanager
  ```

- [x] **3.2 Create docker-compose.yml**
  ```
  File: /home/jlmorg1/monitoring/docker-compose.yml
  What: Prometheus, Grafana, Alertmanager services
  ```

- [x] **3.3 Create prometheus.yml**
  ```
  File: /home/jlmorg1/monitoring/prometheus/prometheus.yml
  What: Scrape configs for node-exporter and cAdvisor
  ```

- [x] **3.4 Create alerts.yml**
  ```
  File: /home/jlmorg1/monitoring/prometheus/alerts.yml
  What: Alert rules for disk, memory, containers
  ```

- [x] **3.5 Create alertmanager config**
  ```
  File: /home/jlmorg1/monitoring/alertmanager/config.yml
  What: Basic alertmanager config (can just log alerts for now)
  ```

- [x] **3.6 Create Grafana datasource**
  ```
  File: /home/jlmorg1/monitoring/grafana/provisioning/datasources/prometheus.yml
  What: Auto-configure Prometheus as data source
  ```

- [x] **3.7 Start monitoring stack**
  ```
  Run: cd /home/jlmorg1/monitoring && docker compose up -d
  Check:
    - Prometheus: http://localhost:9090 ✓
    - Grafana: http://localhost:3001 (admin/admin) ✓
    - Alertmanager: http://localhost:9093 ✓
  ```

---

### Phase 4: Deploy Exporters ✅ COMPLETE

```
Directory: container-management-ansible/ (or just run docker commands)
```

- [x] **4.1 Deploy node-exporter on this server**
  ```
  Running on port 9100
  Test: curl http://localhost:9100/metrics | head ✓
  ```

- [x] **4.2 Deploy cAdvisor on this server**
  ```
  Running on port 8082 (8081 was taken)
  Test: curl http://localhost:8082/metrics | head ✓
  ```

- [x] **4.3 Verify Prometheus sees targets**
  ```
  Check: http://localhost:9090/targets
  All 3 targets showing "UP" ✓
    - prometheus
    - node-aiphdserver1
    - cadvisor-aiphdserver1
  ```

---

### Phase 5: Grafana Dashboards

- [x] **5.1 System Overview dashboard (auto-provisioned)**
  ```
  Custom dashboard created at: /home/jlmorg1/monitoring/grafana/dashboards/system-overview.json
  Includes: CPU gauge, Memory gauge, Disk gauge, Running Containers stat,
            CPU over time, Memory over time
  Access: http://localhost:3001/d/system-overview/system-overview
  ```

- [ ] **5.2 Import Docker dashboard (optional)**
  ```
  In Grafana: Dashboards → Import → ID: 893 → Load → Select Prometheus → Import
  ```

- [ ] **5.3 Create custom User Activity dashboard (optional)**
  ```
  Manual: Create panels showing deployments by user from Streamlit data
  ```

---

## FILE REFERENCE (Copy-Paste Ready)

### models.py location
```
/home/jlmorg1/work/Johnny_AI_ML_LLM_experiments_2024/container-management/streamlit-service/models.py
```

### database.py location
```
/home/jlmorg1/work/Johnny_AI_ML_LLM_experiments_2024/container-management/streamlit-service/database.py
```

### app.py location
```
/home/jlmorg1/work/Johnny_AI_ML_LLM_experiments_2024/container-management/streamlit-service/app.py
```

### monitoring stack location
```
/home/jlmorg1/monitoring/
```

---

## QUICK COMMANDS

```bash
# Go to project
cd /home/jlmorg1/work/Johnny_AI_ML_LLM_experiments_2024/container-management/streamlit-service

# Run the app
streamlit run app.py --server.port 8501

# Check database
sqlite3 ~/.container-deployments.db ".tables"
sqlite3 ~/.container-deployments.db "SELECT * FROM users;"
sqlite3 ~/.container-deployments.db "SELECT * FROM deployments;"

# Start monitoring
cd /home/jlmorg1/monitoring && docker compose up -d

# Check monitoring
docker ps | grep -E "(prometheus|grafana|alertmanager|node-exporter|cadvisor)"

# View logs
docker logs prometheus
docker logs grafana
```

---

## STATUS TRACKER

| Phase | Status | Date | Notes |
|-------|--------|------|-------|
| 1.1 models.py | DONE | 2026-01-27 | Created with User, Deployment, DeploymentStatus, HealthCheck |
| 1.2 database.py | DONE | 2026-01-27 | SQLite with all CRUD operations |
| 1.3 Test database | DONE | 2026-01-27 | Tested user/deployment creation |
| 2.1 Update .env | DONE | 2026-01-27 | Added ENABLE_SIMPLE_AUTH, ADMIN_USERS, DB_PATH |
| 2.2 Update requirements | DONE | 2026-01-27 | pandas installed |
| 2.3 Simple auth | DONE | 2026-01-27 | get_or_select_user() function added |
| 2.4 Database integration | DONE | 2026-01-27 | Deployments saved to SQLite |
| 2.5 Admin Dashboard | DONE | 2026-01-27 | Full admin tab with filters, tables |
| 2.6 Test app | DONE | 2026-01-28 | App running, health OK, DB operational |
| 3.1 Monitoring dirs | DONE | 2026-01-28 | Created at /home/jlmorg1/monitoring/ |
| 3.2 docker-compose | DONE | 2026-01-28 | Prometheus :9090, Grafana :3001, Alertmanager :9093 |
| 3.3 prometheus.yml | DONE | 2026-01-28 | Scrapes node-exporter:9100, cAdvisor:8082 |
| 3.4 alerts.yml | DONE | 2026-01-28 | Disk, memory, CPU, container alerts |
| 3.5 alertmanager | DONE | 2026-01-28 | Basic config with default/critical receivers |
| 3.6 Grafana datasource | DONE | 2026-01-28 | Prometheus auto-provisioned |
| 3.7 Start stack | DONE | 2026-01-28 | All containers running, all targets UP |
| 4.1 node-exporter | DONE | 2026-01-28 | Running on port 9100 |
| 4.2 cAdvisor | DONE | 2026-01-28 | Running on port 8082 (8081 was taken) |
| 4.3 Verify targets | DONE | 2026-01-28 | 3 targets UP: prometheus, node, cadvisor |
| 5.1 Node dashboard | DONE | 2026-01-28 | System Overview dashboard auto-provisioned |
| 5.2 Docker dashboard | NOT STARTED | | Can import ID 893 from Grafana |

---

## FOR CLAUDE IN NEXT SESSION

```
Hey Claude, I'm working on a container management system.
Read this checklist: /home/jlmorg1/work/Johnny_AI_ML_LLM_experiments_2024/container-management/IMPLEMENTATION_CHECKLIST.md
Pick up where we left off. Check the STATUS TRACKER table to see what's done.
The full plan is at: /home/jlmorg1/.claude/plans/distributed-gliding-ember.md
```

---

*"Me fail English? That's unpossible!" - Ralph Wiggum*
*"I'm helping!" - Also Ralph Wiggum (and hopefully this checklist)*
