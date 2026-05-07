// PM2 Ecosystem Config — ZeaCares API
// Usage:
//   pm2 start ecosystem.config.js
//   pm2 restart zeacares-api
//   pm2 logs zeacares-api
//   pm2 stop zeacares-api

const path = require("path");
const projectRoot = __dirname;   // always the directory this file lives in

module.exports = {
  apps: [
    {
      name: "zeacares-api",
      script: "uvicorn",
      args: "src.api.main:app --host 0.0.0.0 --port 8000 --workers 1",
      cwd: projectRoot,           // ← critical: PM2 runs from the project root
      interpreter: path.join(projectRoot, "venv/bin/python3"),
      interpreter_args: "-m",

      // Environment — set absolute paths so nothing depends on cwd
      env: {
        PYTHONPATH: projectRoot,
        RESULTS_DIR: path.join(projectRoot, "results"),
        MODEL_CACHE_DIR: path.join(projectRoot, "model_cache"),
        TRANSFORMERS_CACHE: path.join(projectRoot, "model_cache"),
        HF_HOME: path.join(projectRoot, "model_cache"),
        SENTENCE_TRANSFORMERS_HOME: path.join(projectRoot, "model_cache"),
        DEVICE: "cpu",
        LOG_LEVEL: "INFO",
      },

      // Load .env file for secrets (MONGO_URI, OPENAI_API_KEY)
      // These override the env block above
      env_file: path.join(projectRoot, ".env"),

      // Restart policy — don't restart immediately on port-conflict exit
      autorestart: true,
      max_restarts: 5,
      min_uptime: "10s",
      restart_delay: 3000,

      // Logging
      out_file: path.join(projectRoot, "logs/out.log"),
      error_file: path.join(projectRoot, "logs/error.log"),
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      merge_logs: true,
    },
  ],
};
