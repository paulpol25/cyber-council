# LLM Cybersecurity Council

An AI-powered cybersecurity analysis platform combining:
- **AG2 (AutoGen)** multi-agent collaboration system
- **Machine Learning** models for threat detection
- **MCP (Model Context Protocol)** tools for LLM integration

## 🎯 Overview

The Cyber Council is a **multi-agent AI system** where specialized security experts collaborate to analyze incidents:

### 🤖 AI Agents
- **Incident Commander** - Orchestrates the team
- **Travel Detective** - Detects impossible travel patterns
- **Threat Intel Specialist** - Queries VirusTotal and AbuseIPDB
- **Alert Triager** - Analyzes security alerts
- **Threat Analyst** - Provides overall threat assessment

### 🔧 Capabilities
- **Impossible Travel Detection**: ML-powered detection of physically impossible login patterns
- **Threat Intelligence**: Real-time checks against VirusTotal and AbuseIPDB
- **Security Alert Analysis**: Intelligent triage for 6 alert types
- **Multi-Indicator Assessment**: Aggregate analysis of multiple security indicators
- **Multi-LLM Support**: Works with OpenAI, Google Gemini, and local Ollama models

## 📁 Project Structure

```
llm-council/
├── council/                           # 🤖 AG2 Multi-Agent System
│   ├── orchestrator.py                # Council coordinator
│   ├── agents.py                      # Specialized security agents
│   ├── mcp_adapter.py                 # MCP tool adapter for AG2
│   ├── llm_config.py                  # Multi-LLM configuration
│   ├── cli.py                         # Command-line interface
│   ├── quickstart.py                  # Example scenarios
│   └── README.md                      # Council documentation
│
├── mcp/                               # 🔌 Model Context Protocol Server
│   ├── server.py                      # Main MCP server
│   ├── functions/                     # Security analysis tools
│   │   ├── impossible_travel.py       # ML-powered travel detection
│   │   ├── alert_analyzer.py          # Alert analysis engine
│   │   ├── virustotal_check.py        # VirusTotal integration
│   │   └── abuseipdb_check.py         # AbuseIPDB integration
│   └── README.md                      # MCP server documentation
│
├── models/                            # 🧠 Machine Learning Models
│   └── impossible-travel-detector/    
│       ├── src/                       # Model training and prediction
│       ├── configs/                   # Model configuration
│       ├── data/                      # Training data
│       └── README.md                  # Model documentation
│
├── .env.example                       # Environment configuration template
└── README.md                          # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Install dependencies
python -m venv cyber-council

-- Activate the environment -- 
Linux: source cyber-council/bin/activate
Windows: cybber-council/Scripts/Activate

pip install -r mcp/requirements.txt
pip install -r council/requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Run the Cyber Council

**Interactive Mode:**
```powershell
python council/cli.py interactive
```

**Analyze an Incident:**
```powershell
python council/cli.py analyze "User logged in from US then China in 1 hour"
```

**Quick Checks:**
```powershell
# Check IP reputation
python council/cli.py check-ip 192.0.2.1

# Check file hash
python council/cli.py check-hash 275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f
```

### 3. Train the ML Model (Optional)

For improved impossible travel detection:

```powershell
cd models\impossible-travel-detector
pip install -r requirements.txt
python src\train.py --config configs\config.yaml
```

### 4. Setup MCP Server (Optional)

For direct LLM integration (Claude Desktop, etc.):

```powershell
cd mcp
pip install -r requirements.txt
python setup.py
```

### 3. Test the Functions

```powershell
python test_server.py
```

## 🛠️ Features

### Impossible Travel Detection

Detects when users log in from locations that are physically impossible to reach in the given timeframe.

**Example:**
- Previous login: Tokyo at 2:00 PM
- Current login: New York at 4:00 PM (2 hours later)
- **Result**: IMPOSSIBLE_TRAVEL - Would require 5,425 km/h (faster than any airplane)

### Security Alert Analysis

Analyzes security alerts and provides:
- Threat level assessment (Critical/High/Medium/Low)
- Risk scoring (0-100)
- Indicators of Compromise (IOCs)
- Recommended actions
- Confidence levels

**Supported Alert Types:**
- Authentication anomalies
- Malware detections
- Network intrusions
- Data exfiltration
- Privilege escalation
- Suspicious processes

### Threat Level Assessment

Combines multiple security indicators to provide comprehensive threat assessment:
- Aggregates indicators by severity
- Calculates overall risk score
- Provides prioritized response recommendations
- Determines response urgency level

## 💡 Use Cases

### 1. Multi-Agent Incident Analysis

**Scenario**: Security incident with multiple indicators.

```powershell
python council/cli.py analyze "User alice@company.com logged in from US at 8am, then China at 9am. China IP (203.0.113.42) flagged on threat intel. She accessed financial database during China session."
```

**What happens:**
1. **Incident Commander** receives incident and delegates tasks
2. **Travel Detective** checks impossible travel pattern
3. **Threat Intel Specialist** queries VirusTotal and AbuseIPDB for the IP
4. **Alert Triager** assesses authentication anomaly
5. **Threat Analyst** aggregates all findings
6. **Commander** synthesizes comprehensive incident report

### 2. Quick Threat Intelligence Checks

**Check IP reputation:**
```powershell
python council/cli.py check-ip 192.0.2.1
```

**Check file hash:**
```powershell
python council/cli.py check-hash 275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f
```

**Check domain:**
```powershell
python council/cli.py check-domain evil.com
```

### 3. Interactive Security Analysis

Start interactive mode for ongoing analysis:

```powershell
python council/cli.py interactive
```

Then describe incidents naturally:
```
🚨 Describe the incident:
> Multiple failed login attempts from 10 different IPs in China, 
  then successful login and privileged access granted
```

The council will analyze and provide comprehensive assessment.

### 4. Direct MCP Integration

Use MCP tools with Claude Desktop or other MCP clients for natural language security analysis.

## 🏗️ Architecture

### Multi-Agent System (AG2)

```
┌──────────────────────────────────────────────────────────────┐
│                     Incident Commander                       │
│              (Orchestrates specialist agents)                │
└───────┬──────────────────────────────────────────────────────┘
        │
        │ Delegates tasks
        │
    ┌───┴────────────────────────────────────────────────┐
    │                                                     │
    ▼                  ▼                  ▼              ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Travel  │    │  Threat  │    │  Alert   │    │  Threat  │
│Detective │    │Intel Spec│    │ Triager  │    │ Analyst  │
└────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │               │               │
     │ Uses MCP      │ Uses MCP      │ Uses MCP      │ Uses MCP
     │ Tools         │ Tools         │ Tools         │ Tools
     │               │               │               │
     └───────────────┴───────────────┴───────────────┘
                          │
                          ▼
            ┌─────────────────────────┐
            │     MCP Tool Adapter    │
            └────────────┬────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│impossible_   │  │virustotal_   │  │abuseipdb_    │
│travel.py     │  │check.py      │  │check.py      │
│              │  │              │  │              │
│• ML model    │  │• IP check    │  │• IP reputation│
│• Geo calc    │  │• Hash check  │  │• Abuse score │
└──────────────┘  └──────────────┘  └──────────────┘
```

### LLM Integration

Supports multiple LLM providers:
- **OpenAI GPT-4** - Best performance (recommended)
- **Google Gemini** - Good alternative
- **Ollama** - Local LLMs for privacy (llama2, mistral, etc.)

Each agent can use a different LLM for diverse perspectives.

## 📊 Models

### Impossible Travel Detector

A neural network trained to detect impossible travel patterns based on:
- Geographic distance between login locations
- Time difference between logins
- Required travel speed
- Historical user behavior patterns
- Country and city information

**Performance** (on test set):
- Accuracy: ~95%
- Precision: ~92%
- Recall: ~90%
- F1-Score: ~91%

See `models/impossible-travel-detector/README.md` for details.

## 🧪 Testing

### Run Complete Test Suite

```powershell
python test_council.py
```

### Test Council Examples

```powershell
python council/quickstart.py
```

### Test Individual Components

**MCP Tools:**
```powershell
cd mcp
python examples.py
```

**ML Model:**
```powershell
cd models\impossible-travel-detector
python src\predict.py --username user001 --country "United States" --prev_country Japan --time_diff_hours 2
```

## 📚 Documentation

- **[MCP Server Documentation](mcp/README.md)** - Detailed MCP server guide
- **[Quick Start Guide](mcp/QUICKSTART.md)** - 5-minute setup guide
- **[Model Documentation](models/impossible-travel-detector/README.md)** - ML model details

## 🔐 Security Considerations

- All processing is done locally - no external API calls
- No sensitive data is transmitted outside your environment
- MCP server runs as a local process only
- Ensure proper file permissions on config files
- Review all recommendations before taking action

## 🛣️ Roadmap

- [ ] Add more ML models (phishing detection, anomaly detection)
- [ ] Integration with real threat intelligence feeds
- [ ] Support for more MCP clients
- [ ] Web dashboard for visualizations
- [ ] Real-time alert streaming
- [ ] Historical pattern analysis
- [ ] Automated response actions
- [ ] SIEM integration

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built with [Model Context Protocol](https://modelcontextprotocol.io/)
- Uses TensorFlow/Keras for ML models
- Inspired by real-world SOC analyst workflows

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation in each component's README
- Review the examples for usage patterns

---

**Made with ❤️ for cybersecurity professionals and AI enthusiasts**
