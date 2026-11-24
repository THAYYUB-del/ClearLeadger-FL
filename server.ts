/**
 * ClearLedger-FL Full-Stack Express Server.
 * Serves all 19 REST API endpoints backed by real Logistic Regression, FedAvg,
 * HMAC-SHA256 verification, SHA-256 hash-linked SQLite audit ledger, and exact Linear SHAP,
 * and mounts Vite middlewares (dev) or static dist/ (production) on port 3000.
 */

import 'dotenv/config';
import express, { type Request, type Response } from 'express';
import fs from 'node:fs';
import path from 'node:path';
import { createServer as createViteServer } from 'vite';
import {
  FederatedCoordinator,
  type SecurityScenario,
} from './server/coordinator';
import type { DistributionMode, HospitalId } from './server/dataset';

const VALID_HOSPITALS = new Set<HospitalId>(['hospital_a', 'hospital_b', 'hospital_c']);
const VALID_SCENARIOS = new Set<SecurityScenario>([
  'valid_update',
  'invalid_hmac',
  'forged_sender',
  'exaggerated_update',
  'label_flipping',
]);

function parseHospitalId(raw: string): HospitalId | null {
  return VALID_HOSPITALS.has(raw as HospitalId) ? (raw as HospitalId) : null;
}

export function createApiApp(coordinator: FederatedCoordinator) {
  const app = express();
  app.use(express.json({ limit: '1mb' }));

  // 1. Health endpoint
  app.get('/api/health', (_req: Request, res: Response) => {
    res.json({
      status: 'ok',
      service: 'ClearLedger-FL API',
      timestamp: new Date().toISOString(),
      database: coordinator.db.dbPath === ':memory:' ? 'sqlite-memory' : 'sqlite-file',
    });
  });

  // 2. Overview Dashboard
  app.get('/api/dashboard', (_req: Request, res: Response) => {
    res.json(coordinator.getDashboardSummary());
  });

  // 3. List Hospitals
  app.get('/api/hospitals', (_req: Request, res: Response) => {
    res.json({
      hospitals: coordinator.getHospitals(),
    });
  });

  // 4. Inspect single Hospital
  app.get('/api/hospitals/:hospital_id', (req: Request, res: Response) => {
    const hid = parseHospitalId(String(req.params.hospital_id));
    if (!hid) {
      res.status(404).json({ detail: 'Hospital node not found. Expected hospital_a, hospital_b, or hospital_c.' });
      return;
    }
    res.json(coordinator.getHospitalDetail(hid));
  });

  // 5. Initialize Experiment
  app.post('/api/experiment/initialize', (req: Request, res: Response) => {
    try {
      const body = req.body ?? {};
      const distribution: DistributionMode =
        body.distribution === 'non_iid' ? 'non_iid' : 'balanced';
      const seed = body.seed !== undefined ? Number(body.seed) : 42;
      const local_epochs = body.local_epochs !== undefined ? Number(body.local_epochs) : 12;
      const learning_rate = body.learning_rate !== undefined ? Number(body.learning_rate) : 0.12;

      if (!Number.isFinite(seed) || seed < 1 || seed > 999999) {
        res.status(400).json({ detail: 'seed must be an integer between 1 and 999999.' });
        return;
      }
      if (!Number.isFinite(local_epochs) || local_epochs < 1 || local_epochs > 50) {
        res.status(400).json({ detail: 'local_epochs must be between 1 and 50.' });
        return;
      }
      if (!Number.isFinite(learning_rate) || learning_rate <= 0 || learning_rate > 2.0) {
        res.status(400).json({ detail: 'learning_rate must be between 0.005 and 2.0.' });
        return;
      }

      const status = coordinator.initializeExperiment({
        distribution,
        seed,
        local_epochs,
        learning_rate,
      });
      res.json(status);
    } catch (err) {
      res.status(400).json({ detail: err instanceof Error ? err.message : 'Initialization failed.' });
    }
  });

  // 6. Local Hospital Training
  app.post('/api/hospitals/:hospital_id/train', (req: Request, res: Response) => {
    const hid = parseHospitalId(String(req.params.hospital_id));
    if (!hid) {
      res.status(404).json({ detail: 'Hospital node not found.' });
      return;
    }
    try {
      const epochs = req.body?.local_epochs !== undefined ? Number(req.body.local_epochs) : undefined;
      const result = coordinator.trainSingleHospital(hid, epochs);
      res.json(result);
    } catch (err) {
      res.status(400).json({ detail: err instanceof Error ? err.message : 'Local training failed.' });
    }
  });

  // 7. Start Federated Training
  app.post('/api/federated/start', (req: Request, res: Response) => {
    try {
      const rounds = req.body?.rounds !== undefined ? Number(req.body.rounds) : 1;
      if (!Number.isFinite(rounds) || rounds < 1 || rounds > 20) {
        res.status(400).json({ detail: 'rounds must be an integer between 1 and 20.' });
        return;
      }
      const result = coordinator.runFederatedTraining(rounds);
      res.json(result);
    } catch (err) {
      res.status(400).json({ detail: err instanceof Error ? err.message : 'Federated training failed.' });
    }
  });

  // 8. Federated Status
  app.get('/api/federated/status', (_req: Request, res: Response) => {
    res.json(coordinator.getFederatedStatus());
  });

  // 9. Federated Rounds History
  app.get('/api/federated/rounds', (_req: Request, res: Response) => {
    res.json({
      rounds: coordinator.db.getRounds(),
    });
  });

  // 10. Reset Experiment
  app.post('/api/experiment/reset', (_req: Request, res: Response) => {
    res.json(coordinator.resetExperiment());
  });

  // 11. Security Events
  app.get('/api/security/events', (_req: Request, res: Response) => {
    const events = coordinator.db.getSecurityEvents(200);
    const accepted = events.filter((e) => e.decision === 'ACCEPTED').length;
    const rejectedAuth = events.filter((e) => e.layer_failed === 'AUTHENTICATION').length;
    const rejectedValidation = events.filter(
      (e) => e.layer_failed === 'MALICIOUS_VALIDATION'
    ).length;

    res.json({
      summary: {
        total_events: events.length,
        accepted,
        rejected_total: rejectedAuth + rejectedValidation,
        rejected_authentication: rejectedAuth,
        rejected_malicious_validation: rejectedValidation,
      },
      events,
    });
  });

  // 12. Simulate Security / Malicious Update Experiment
  app.post('/api/security/simulate', (req: Request, res: Response) => {
    const scenario = String(req.body?.scenario ?? '') as SecurityScenario;
    const hospitalId = parseHospitalId(String(req.body?.hospital_id ?? 'hospital_b'));

    if (!VALID_SCENARIOS.has(scenario)) {
      res.status(400).json({
        detail:
          'Invalid scenario. Expected one of: valid_update, invalid_hmac, forged_sender, exaggerated_update, label_flipping.',
      });
      return;
    }
    if (!hospitalId) {
      res.status(400).json({ detail: 'Invalid hospital_id.' });
      return;
    }

    try {
      const result = coordinator.simulateSecurityExperiment(scenario, hospitalId);
      res.json(result);
    } catch (err) {
      res.status(400).json({ detail: err instanceof Error ? err.message : 'Security test failed.' });
    }
  });

  // 13. Get Hash-Linked Audit Ledger
  app.get('/api/ledger', (_req: Request, res: Response) => {
    res.json(coordinator.getLedger());
  });

  // 14. Verify Audit Ledger Integrity
  app.post('/api/ledger/verify', (_req: Request, res: Response) => {
    res.json(coordinator.verifyLedger());
  });

  // 15. Controlled Non-Destructive Tampering Demonstration
  app.post('/api/ledger/tamper-test', (req: Request, res: Response) => {
    try {
      const recordId =
        req.body?.target_record_id !== undefined
          ? Number(req.body.target_record_id)
          : undefined;
      const result = coordinator.runTamperTest(
        recordId !== undefined && Number.isFinite(recordId) ? recordId : undefined
      );
      res.json(result);
    } catch (err) {
      res.status(400).json({ detail: err instanceof Error ? err.message : 'Tamper test failed.' });
    }
  });

  // 16. Global SHAP Explainability
  app.get('/api/explainability/global', (_req: Request, res: Response) => {
    res.json(coordinator.getGlobalExplainability());
  });

  // 17. Individual SHAP Prediction Explanation
  app.post('/api/explainability/predict', (req: Request, res: Response) => {
    try {
      const sampleIndex = Number(req.body?.sample_index ?? 0);
      const explanation = coordinator.explainPrediction(sampleIndex);
      res.json(explanation);
    } catch (err) {
      res.status(400).json({ detail: err instanceof Error ? err.message : 'SHAP explanation failed.' });
    }
  });

  // 18. Model Analytics
  app.get('/api/analytics', (_req: Request, res: Response) => {
    res.json(coordinator.getAnalytics());
  });

  // 19. Export Model Analytics as CSV
  app.get('/api/analytics/export', (_req: Request, res: Response) => {
    try {
      const csv = coordinator.exportAnalyticsCsv();
      res.setHeader('Content-Type', 'text/csv; charset=utf-8');
      res.setHeader(
        'Content-Disposition',
        'attachment; filename="clearledger-fl-analytics.csv"'
      );
      res.send(csv);
    } catch (err) {
      res.status(400).json({ detail: err instanceof Error ? err.message : 'CSV export unavailable.' });
    }
  });

  return app;
}

async function startServer() {
  const dbPath = process.env.CLEARLEDGER_DB_PATH
    ? path.resolve(process.cwd(), process.env.CLEARLEDGER_DB_PATH)
    : path.resolve(process.cwd(), 'results', 'clearledger.db');

  const coordinator = new FederatedCoordinator(dbPath);
  const app = createApiApp(coordinator);

  const isProd = process.env.NODE_ENV === 'production';
  const distDir = path.resolve(process.cwd(), 'dist');

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else if (fs.existsSync(distDir)) {
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  const PORT = Number(process.env.PORT ?? 3000);
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`ClearLedger-FL server running on http://localhost:${PORT}`);
  });
}

// Only bind port when run directly (not when imported by unit tests)
const isDirectExecution =
  process.argv[1] &&
  path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname);

if (isDirectExecution) {
  startServer().catch((err) => {
    console.error('Failed to start ClearLedger-FL server:', err);
    process.exit(1);
  });
}
