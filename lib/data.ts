// Static data for projects and categories
// This file contains all the project and category data that can be easily managed

export interface Project {
  id: string
  title: string
  description: string
  image?: string
  technologies: string
  github?: string
  demo?: string
  featured: boolean
  published: boolean
  categoryId: string
  createdAt: string
  updatedAt: string
}

export interface Category {
  id: string
  name: string
  description: string
  color: string
  icon: string
  createdAt: string
  updatedAt: string
}

// Categories data
export const categories: Category[] = [
  {
    id: '1',
    name: 'Deep Learning',
    description: 'Design and integration of Deep Learning models with deployment, monitoring, and maintenance as microservices and cloud services.',
    color: '#3B82F6', // Blue
    icon: '🎨',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '2',
    name: 'Time Series Analysis and Forecasting',
    description: 'Data preprocessing, augmentation, feature engineering, model selection, training, evaluation, deployment and insight dashboard creation for time series data.',
    color: '#10B981', // Green
    icon: '⚙️',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '3',
    name: 'Multi-Modal Learning',
    description: 'LLMs, VLLMs, and other multi-modal models for various applications including chatbots, content generation, image recognition, and more.',
    color: '#8B5CF6', // Purple
    icon: '🚀',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '4',
    name: 'Knowledge Distillation',
    description: 'Knowledge distillation techniques to create smaller, faster, and more efficient models without significant loss in performance.',
    color: '#F59E0B', // Orange
    icon: '📱',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '5',
    name: 'Reinforcement Learning',
    description: 'Reinforcement learning algorithms and applications for decision-making, game playing, robotics, industrial automation, personalized recommendations, and many more.',
    color: '#EF4444', // Red
    icon: '🔌',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '6',
    name: 'Federated Learning',
    description: 'Federated learning techniques for training models across decentralized devices or servers while preserving data privacy and security.',
    color: '#06B6D4', // Cyan
    icon: '🛠️',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '7',
    name: 'Multimedia AI',
    description: 'Multimedia AI techniques for processing and analyzing various types of media data including images, videos, audio, and text.',
    color: '#EC4899', // Pink
    icon: '🤖',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '8',
    name: 'Machine Learning Operations (MLOps)',
    description: 'MLOps practices for managing the end-to-end machine learning lifecycle including model development, deployment, monitoring, and maintenance.',
    color: '#84CC16', // Lime
    icon: '⛓️',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '9',
    name: 'Quantum Machine Learning (QML)',
    description: 'Quantum machine learning algorithms and applications for leveraging quantum computing to enhance machine learning tasks.',
    color: '#6B7280', // Gray
    icon: '🔧',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '10',
    name: 'Edge AI',
    description: 'Edge AI techniques for deploying machine learning models on edge devices for real-time inference and decision-making.',
    color: '#8B5CF6', // Purple
    icon: '📊',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  }
]

// Projects data
export const projects: Project[] = [
    {
    id: '-1',
    title: 'Real-Time AI for Autonomous Defect Detection in Infrastructure (Drone-Integrated)',
    description: 'Develop an end-to-end system where a drone autonomously flies a pre-planned route to inspect large industrial structures such as bridges, wind turbines, or power lines.',
    image: '/AI_Portfolio/dl/dl-5.png',
    technologies: 'YOLOv8-Nano, EfficientDet, DJI SDK, PX4/ArduPilot, Python, NVIDIA TensorRT, MAVLink',
    // github: 'https://github.com/example/ecommerce',
    // demo: 'https://ecommerce-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '1', // Deep Learning
    createdAt: '2024-01-15T00:00:00Z',
    updatedAt: '2024-01-15T00:00:00Z',
  },
  {
    id: '0',
    title: 'AI-Driven Health Assessment and Predictive Maintenance for Telecom Towers',
    description: 'The goal is to automatically analyze drone-captured imagery of a telecom tower to assess its structural health, component integrity, and compliance.',
    image: '/AI_Portfolio/dl/dl-4.png',
    technologies: 'Mask R-CNN, YOLO-v8 Segment, OpenCV, Azure, GDAL',
    // github: 'https://github.com/example/ecommerce',
    // demo: 'https://ecommerce-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '1', // Deep Learning
    createdAt: '2024-01-15T00:00:00Z',
    updatedAt: '2024-01-15T00:00:00Z',
  },
  {
    id: '1',
    title: 'Automated Property Feature Extraction and Valuation from Aerial Imagery',
    description: 'The system uses advanced computer vision techniques to analyze aerial photos of residential and commercial properties.',
    image: '/AI_Portfolio/dl/dl-3.jpg',
    technologies: 'DeepLabV3+, ArcGIS, Rasterio, Geopandas, XGBoost, Google Earth Engine API, Streamlit, Python, OpenCV, Scikit-image, FastAPI, Trasnfer Learning, Object Detection, Image Classification',
    // github: 'https://github.com/example/ecommerce',
    // demo: 'https://ecommerce-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '1', // Deep Learning
    createdAt: '2024-01-15T00:00:00Z',
    updatedAt: '2024-01-15T00:00:00Z',
  },
  {
    id: '2',
    title: 'Historical Blueprints & Engineering Drawing Digitizer',
    description: 'This custom OCR system is designed to digitize and extract critical information from old, faded, or sepia-toned engineering drawings and blueprints. It accurately identifies and extracts handwritten annotations, dimensions, component labels, and revision numbers, converting them into searchable, structured digital data. This preserves valuable historical information, prevents loss due to physical degradation, and enables efficient search and integration with modern CAD/PLM systems.',
    image: '/AI_Portfolio/dl/dl-1.png',
    technologies: 'Azure VMs, PyTorch, PostgreSQL, TensorFlow, PaddleOCR, Tesseract OCR, Streamlit, Python, OpenCV, Scikit-image, FastAPI, Trasnfer Learning, Object Detection, Image Classification',
    // github: 'https://github.com/example/ecommerce',
    // demo: 'https://ecommerce-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '1', // Deep Learning
    createdAt: '2024-01-15T00:00:00Z',
    updatedAt: '2024-01-15T00:00:00Z',
  },
  {
    id: '3',
    title: 'Damaged Freight Label Processor',
    description: 'Never lose a package to a bad label again. This specialized system is designed to rescue data from compromised shipping documents, whether they are wrinkled, smudged, or ripped. By accurately isolating and reading damaged text and barcodes, it provides uninterrupted data flow for inventory management and package routing, even under poor conditions.',
    image: '/AI_Portfolio/dl/dl-2.png',
    technologies: 'Python, OpenCV, PyTorch, Customly Trained Hybrid Transformer, Scikit-image, PaddleOCR, Custom Data, Offline Custom Server for Deployment, FastAPI, Object Detection, Image Classification',
    // github: 'https://github.com/example/task-manager',
    // demo: 'https://task-manager-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '1', // Deep Learning
    createdAt: '2024-01-10T00:00:00Z',
    updatedAt: '2024-01-10T00:00:00Z',
  },
  {
    id: '4',
    title: 'Automated Inventory Demand Predictor',
    description: 'Predicts exactly how much product customers will buy at different times and locations.',
    image: '/AI_Portfolio/time-series/time-series-4.jpeg',
    technologies: 'FastAPI, Custom Dashboard with Streamlit, Custom service integrated with email, AWS Lambda, AWS Step Functions, Docker, MLflow, Exponential Smoothing, Auto-ARIMA, Prophet from Meta, CatBoost, TimescaleDB, Warehouse Management Systems',
    // github: 'https://github.com/example/weather-dashboard',
    // demo: 'https://weather-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '2', // Time Series
    createdAt: '2024-01-05T00:00:00Z',
    updatedAt: '2024-01-05T00:00:00Z',
  },
  {
    id: '5',
    title: 'Predictive Maintenance Platform',
    description: 'Uses time series sensor data from critical machines to forecast RUL.',
    image: '/AI_Portfolio/time-series/time-series-3.jpeg',
    technologies: 'Python 3.10, Pandas, Scikit-learn, MLflow, LSTMs, GRUs, TensorFlow, PyTorch, Gradient Boosting, Grafana, Django, React, Amazon S3',
    // github: 'https://github.com/example/cms-admin',
    // demo: 'https://cms-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '2', // Time Series
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '6',
    title: 'Real-time Stock Market Prediction',
    description: 'Real-time stock market Prediction for managing stock profile. Features include live data streaming, predictive analytics, and portfolio optimization tools.',
    image: '/AI_Portfolio/time-series/time-series-2.png',
    technologies: 'Python 3.9, ARIMA, CatBoost, LightGBM, Flask, Docker, Postgres',
    // github: 'https://github.com/example/ai-chat',
    // demo: 'https://ai-chat-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '2', // Time Series
    createdAt: '2024-02-01T00:00:00Z',
    updatedAt: '2024-02-01T00:00:00Z',
  },
  {
    id: '7',
    title: 'Forecasting Application for Fisheries Production',
    description: 'Prediction of fisheries production to provide insights on fisheries economy.',
    image: '/AI_Portfolio/time-series/time-series-1.jpeg',
    technologies: 'Python 3.9, Prophet, SARIMAX, XGBoost, FastAPI, Docker',
    // github: 'https://github.com/example/blockchain-voting',
    // demo: 'https://voting-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '2', // Time Series
    createdAt: '2024-02-05T00:00:00Z',
    updatedAt: '2024-02-05T00:00:00Z',
  },
  {
    id: '8',
    title: 'Mobile-based Custom Product Recognition',
    description: 'Distill a large image classification model for deployment in a mobile app, allowing users to instantly identify products by taking a picture. This improves the customer experience with less lag time.',
    image: '/AI_Portfolio/distill/distil-1.png',
    technologies: 'PyTorch, TFLite, ONNX, Android Studio, Python, MobileNet',
    // github: 'https://github.com/example/ml-dashboard',
    // demo: 'https://ml-dashboard.vercel.app',
    featured: false,
    published: true,
    categoryId: '4', // 'Knowledge Distillation'
    createdAt: '2024-02-10T00:00:00Z',
    updatedAt: '2024-02-10T00:00:00Z',
  },
  {
    id: '9',
    title: 'Edge-Based Predictive Maintenance for Industrial IoT',
    description: 'The project, Edge-Based Predictive Maintenance for Industrial IoT, solves the challenge of slow and costly cloud analysis of massive industrial sensor data by employing Knowledge Distillation. It shrinks a complex, multi-modal Teacher model (trained on vibration, temperature, and current data) into a tiny Student model that runs locally on an edge micro-controller. This allows for instantaneous anomaly detection directly on the machine, which drastically minimizes downtime by providing immediate failure alerts and cuts data costs by only sending necessary alert data to the cloud.',
    image: '/AI_Portfolio/distill/distill-2.png',
    technologies: 'TinyML, MQTT, ONNX, Model Quantization, RNNs, TCNs, Transformers, TFLite Micro, MicroPython, AWS S3',
    // github: 'https://github.com/example/iot-home',
    // demo: 'https://iot-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '4', // 'Knowledge Distillation'
    createdAt: '2024-02-15T00:00:00Z',
    updatedAt: '2024-02-15T00:00:00Z',
  },
  {
    id: '10',
    title: 'Low-Latency NLP Agent for Contact Centers',
    description: 'This project develops a Real-Time Customer Sentiment Analyzer for contact centers using Knowledge Distillation. We take a slow, powerful Teacher LLM and compress its analytical expertise into a compact, specialized Student model (like DistilBERT). This highly efficient agent delivers sub-50ms predictions for intent and sentiment directly to the human agent, providing instant, actionable insights. The result is faster issue resolution, lower customer churn, and significantly reduced operational costs compared to running massive cloud models.',
    image: '/AI_Portfolio/distill/distil-3.png',
    technologies: 'TinyML, MQTT, ONNX, Model Quantization, RNNs, TCNs, Transformers, TFLite Micro, MicroPython, AWS S3', 
    // github: 'https://github.com/example/crypto-tracker',
    // demo: 'https://crypto-tracker.vercel.app',
    featured: false,
    published: true,
    categoryId: '4', // 'Knowledge Distillation'
    createdAt: '2024-02-20T00:00:00Z',
    updatedAt: '2024-02-20T00:00:00Z',
  },
  {
    id: '11',
    title: 'Cross-Lingual Legal Document Summarizer',
    description: 'Legal experts often spend days analyzing contracts, regulations, and patents written in foreign languages. Generic translation is insufficient; they need accurate, concise summaries and comparisons of specific legal clauses across jurisdictions. The product is an automated legal summarization and clause-comparison tool designed for international law firms, global compliance departments, and multinational corporations (MNCs).',
    image: '/AI_Portfolio/distill/distil-4.png',
    technologies: 'Large Multilingual Language Models (MLMs), mBART, mT5, XLM-R, BERT, RoBERTa, TextRank, MMR, RAG, Cross-Lingual Knowledge Distillation, Docker, HIPAA, GDPR',
    // github: 'https://github.com/example/social-analytics',
    // demo: 'https://social-analytics.vercel.app',
    featured: false,
    published: true,
    categoryId: '4', // 'Knowledge Distillation'
    createdAt: '2024-02-25T00:00:00Z',
    updatedAt: '2024-02-25T00:00:00Z',
  },
  {
    id: '12',
    title: "Vertical Federated Learning for Cross-Company Supply Chain Risk Forecasting",
    description: "This project uses Vertical Federated Learning, known as VFL, to let different, independent companies in a supply chain such as a supplier, a manufacturer, and a distributor collaboratively build a single smart model to predict risks like delays or quality failures.",
    image: '/AI_Portfolio/fedl/fedl-1.png',
    technologies: 'FATE (Federated AI Technology Enabler), OpenFL, Python, Docker, DNN, XGBoost, AWS',
    // github: 'https://github.com/example/real-estate',
    // demo: 'https://real-estate-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '6', // 'Federated Learning'
    createdAt: '2024-03-01T00:00:00Z',
    updatedAt: '2024-03-01T00:00:00Z',
  },
  {
    id: '13',
    title: 'Federated Quality Control for Multi-Vendor Manufacturing Line',
    description: 'Federated Quality Control (FQC) leverages Federated Learning (FL) to enable competing manufacturing equipment vendors to collectively enhance a shared AI quality model. Vendors locally train the model on their proprietary data such as defect images and only securely exchange the model updates. This method yields a superior, global quality model for the manufacturer while fully preserving each vendor\'s data privacy and competitive IP.',
    image: '/AI_Portfolio/fedl/fedl-2.png',
    technologies: 'Docker, Python, PyGrid, PySyft, FATE (Federated AI Technology Enabler), OpenFL, CNNs, Transfer Learning, AWS',
    // github: 'https://github.com/example/fitness-tracker',
    // demo: 'https://fitness-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '6', // 'Federated Learning'
    createdAt: '2024-03-05T00:00:00Z',
    updatedAt: '2024-03-05T00:00:00Z',
  },
  {
    id: '14',
    title: 'Personalized Federated Learning for Multi-Factory Energy Optimization',
    description: 'Leveraging Personalized Federated Learning (pFL), this project dramatically boosts energy efficiency across a corporation\'s diverse factories. Despite heterogeneous data (Non-IID), the system enables each plant to locally train an AI model on its private energy data. By securely aggregating common energy-saving patterns into collective intelligence, pFL then generates a personalized AI model tailored for each factory\'s specific conditions. This delivers maximum cost savings and optimized resource allocation while ensuring data privacy and preserving local operational autonomy.',
    image: '/AI_Portfolio/fedl/fedl-3.png',
    technologies: 'Flower, FedML, FedProx, pFedMe, Docker, MicroK8s, InfluxDB, MQTT, RNNs, TCNs, TLS/VPN',
    // github: 'https://github.com/example/rest-api',
    // demo: 'https://api-docs.vercel.app',
    featured: false,
    published: true,
    categoryId: '6', // 'Federated Learning'
    createdAt: '2024-03-10T00:00:00Z',
    updatedAt: '2024-03-10T00:00:00Z',
  },
  {
    id: '15',
    title: 'Federated Anomaly Detection for Collaborative Industrial Cybersecurity (OT/IIoT)',
    description: 'This project uses Federated Learning (FL) to enable multiple, independent industrial plants to collaboratively train a powerful threat detection model for their OT/IIoT networks. Each plant keeps its sensitive operational data and network traffic private on-site. The FL system securely aggregates only the lessons-learned (model updates) from all sites, creating a superior, collective defense against cyber threats and anomalies without compromising any single plant\'s security or IP.',
    image: '/AI_Portfolio/fedl/fedl-4.png',
    technologies: 'NVIDIA FLARE, Flower, Python, Docker, Autoencoders, One-Class SVM, AWS, Differential Privacy (DP), AzurDevops, MQTT, HTTP',
    // github: 'https://github.com/example/wordpress-store',
    // demo: 'https://wordpress-store.vercel.app',
    featured: false,
    published: true,
    categoryId: '6', // 'Federated Learning'
    createdAt: '2024-03-15T00:00:00Z',
    updatedAt: '2024-03-15T00:00:00Z',
  },
  {
    id: '16',
    title: 'Dynamic Inventory and Production Planning for a Supply Chain',
    description: 'This project is a Multi Agent Reinforcement Learning (MARL) system designed to automatically optimize complex supply chains. It uses cooperating AI agents, a Production Agent at the factory and Distribution Agents at the warehouses, to make real time decisions on what to produce and where to ship it. The goal is to minimize total operational costs production holding and transport while maximizing customer fulfillment avoiding stockouts under volatile demand.',
    image: '/AI_Portfolio/rl/rl-1.jpeg',
    technologies: 'PyTorch, Stable Baselines3, Pandas, Seaborn, AWS Step Functions, Docker, Flask, Typescript',
    // github: 'https://github.com/example/shopify-store',
    // demo: 'https://shopify-store.myshopify.com',
    featured: false,
    published: true,
    categoryId: '5', // Reinforcement Learning
    createdAt: '2024-03-20T00:00:00Z',
    updatedAt: '2024-03-20T00:00:00Z',
  },
  {
    id: '17',
    title: 'Adaptive Process Control for Chemical/Manufacturing Plants',
    description: 'This project develops a Reinforcement Learning RL agent to serve as a self learning advanced controller for complex industrial equipment. Unlike rigid traditional controllers, the RL agent dynamically adjusts system parameters like temperature and flow in real time. The agent is trained to maximize product quality and yield while minimizing energy consumption and strictly adhering to all safety constraints. The final deliverable is a high fidelity simulation demonstrating the RL agent\'s superior efficiency and throughput compared to a benchmark control system.',
    image: '/AI_Portfolio/rl/rl-2.jpg',
    technologies: 'PyTorch, Ray RLlib, Python, Plotly, Azure DevOps, Docker, NumPy, FastAPI',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',
    featured: false,
    published: true,
    categoryId: '5', // Reinforcement Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  {
    id: '18',
    title: 'Conveyor Speed Optimization by RL Agent in Real Time',
    description: 'This project tackles a common industrial control dilemma: finding the sweet spot between saving energy and boosting production. We use foundational RL algorithms to train a smart system to dynamically manage conveyor belt speed in real-time, ensuring we get the maximum amount of product moved with the minimum possible energy usage.',
    image: '/AI_Portfolio/rl/rl-3.png',
    technologies: 'Q-Learning, SARSA, Python, Physics, Simulation',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',
    featured: false,
    published: true,
    categoryId: '5', // Reinforcement Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
   {
    id: '19',
    title: 'RL-Powered Personal E-Commerce Navigator',
    description: 'Traditional e-commerce optimization (like simple A/B testing or static recommendation engines) focuses on maximizing the immediate click or purchase. This RL approach introduces a long-term goal: the agent learns that sometimes, not showing a high-value product right now, or even offering a small discount, is the optimal action because it builds trust, prevents customer burnout, and maximizes the customer\'s total spending over the next year (CLV).',
    image: '/AI_Portfolio/rl/rl-4.png',
    technologies: 'Deep Q-Network (DQN), Policy Gradient (A2C/A3C), Python, Simulation',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',
    featured: false,
    published: true,
    categoryId: '5', // Reinforcement Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
      {
    id: '20',
    title: 'Multi-Sensory Anomaly Detection for Industrial Equipment',
    description: 'This project aims to develop a robust predictive maintenance system for industrial machinery by integrating and analyzing multimodal sensor data.',
    image: '/AI_Portfolio/multi/multi-1.jpg',
    technologies: 'FLIR thermal cameras, Industrial microphones, NVIDIA Jetson Nano Super, AWS S3, InfluxDB, Prometheus, PyTorch, Docker, Grafana, Transformers, CNNs, RNNs, Fusion Models',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
      {
    id: '21',
    title: 'AI-Powered Multimodal Quality Inspection for Assembly Lines',
    description: 'This project focuses on enhancing quality control in manufacturing processes, particularly for assembly tasks, by leveraging computer vision and force/pressure sensor data.',
    image: '/AI_Portfolio/multi/multi-2.png',
    technologies: 'Industrial cameras - Basler, Structured Light Scanners for 3D Inspection, Force/torque Sensors, PyTorch, LSTMs, CNNs, Fusion Modeling, Edge AI Devices, Grafana, SQL',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
   {
    id: '22',
    title: 'Context-Aware Multimodal Assistant for Collaborative Robots',
    description: 'This project aims to create a more intuitive and safe human-robot collaboration (HRC) environment in logistics and warehousing.',
    image: '/AI_Portfolio/multi/multi-3.jpg',
    technologies: 'RGB-D Cameras, Cloud storage, ROS Bag Files for Robot Data, OpenPose, TensorFlow, Transformers, GNNs, Attention-based Fusion, Joint Embeddings, NLU, ROS, Specialized Embedded AI Hardware',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
   {
    id: '23',
    title: 'Multilingual LLM Chatbot for Customer Support',
    description: 'Created a real-time multilingual LLM chatbot capable of handling support queries in 15+ languages with sentiment-aware response tuning.',
    image: '/AI_Portfolio/llm/l5.png',
    technologies: 'Python 3.8, PyTorch, MarianMT, Redis, FastAPI, Kubernetes',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
     {
    id: '24',
    title: 'RAG-Based Enterprise Knowledge Assistant',
    description: 'Built a context-aware RAG assistant to answer internal employee queries by grounding LLM responses in enterprise documents, policies, and knowledge bases.',
    image: '/AI_Portfolio/llm/l4.png',
    technologies: 'Python 3.9, PyTorch, Hugging Face Transformers, LangChain, FAISS, Pinecone, FastAPI, Docker',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
      {
    id: '25',
    title: 'LLM-based Content Generation Engine',
    description: 'Engineered a scalable content generation pipeline for SEO blogs, product descriptions, and brand copy using GPT-J and T5 models.',
    image: '/AI_Portfolio/llm/l6.png',
    technologies: 'Python 3.8, Hugging Face Transformers, GPT-J, T5, Gradio, Docker',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  {
    id: '26',
    title: 'SmartLegalBot - AI Legal Assistant for Contract Review',
    description: 'SmartLegalBot is an AI-powered assistant that helps users review legal contracts, highlight risky clauses, suggest simplified rewrites, and answer basic legal queries, all using a fine-tuned LLM trained on legal documents.',
    image: '/AI_Portfolio/ai/llm.png',
    technologies: 'Node.js, Hugging Face Transformers, PostgreSQL',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
    {
    id: '27',
    title: 'Custom LLM Development',
    description: 'Developed a custom large language model from scratch, optimized for enterprise applications. The model was trained on domain-specific data and fine-tuned for various business use cases, achieving state-of-the-art performance while maintaining efficiency.',
    image: '/AI_Portfolio/ai/llm1.png',
    technologies: 'Python 3.8+, PyTorch, DeepSpeed, PEFT, LoRA, FastAPI, Docker, NVIDIA A100 GPUs',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
      {
    id: '28',
    title: 'LLM Fine-tuning Framework',
    description: 'Created a comprehensive framework for fine-tuning large language models efficiently. The framework supports various fine-tuning methods, including LoRA, PEFT, and full fine-tuning, with automated optimization and monitoring capabilities.',
    image: '/AI_Portfolio/ai/llm2.png',
    technologies: 'Python 3.8+, PyTorch, Transformers, DeepSpeed, PEFT, LoRA, FastAPI, Docker, AWS',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
      {
    id: '29',
    title: 'Autonomous Research Agent using LLMs',
    description: 'Developed an autonomous LLM agent for literature reviews with semantic paper search, PDF summarization, and research graph generation.',
    image: '/AI_Portfolio/llm/l7.png',
    technologies: 'Python 3.10, LangChain, Transformers, PyMuPDF, Neo4j',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
    {
    id: '30',
    title: 'LLM-Powered Code Generation & Review Assistant',
    description: 'Built an AI coding assistant that generates boilerplate code, reviews PRs, and flags anti-patterns with LLM-powered insights.',
    image: '/AI_Portfolio/llm/l9.png',
    technologies: 'Python 3.10, CodeLLaMA, GitHub API, FastAPI, Docker',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  
   {
    id: '31',
    title: 'Quantum-Enhanced Financial Risk Modeling',
    description: 'Developing a hybrid quantum-classical model to enhance the accuracy and speed of financial risk calculations such as value at risk, portfolio optimization by using quantum circuits for complex, high-dimensional probability distributions.',
    image: '/AI_Portfolio/qml/qml-1.png',
    technologies: 'Qiskit, Cirq, Pennylane, TensorFlow Quantum, D-Wave',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '9', // Quantum Machine Learning (QML)
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  {
    id: '32',
    title: 'Quantum Circuit Optimization for Drug Discovery',
    description: 'Implementing QML algorithms to simulate molecular interactions and optimize the design of novel drug compounds, significantly reducing the computational time compared to classical methods for small molecules.',
    image: '/AI_Portfolio/qml/qml-2.png',
    technologies: 'PyQuil, NumPy, scikit-learn, Quantum Chemistry Libraries Quantum, Quantum Cloud Services',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '9', // Quantum Machine Learning (QML)
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
   {
    id: '33',
    title: 'Deepfake Detection and Integrity Verification for Industrial Compliance',
    description: 'Develop an industrial system to automatically detect and flag deepfake content (video, audio, or images) that could compromise safety, security, or regulatory compliance. ',
    image: '/AI_Portfolio/mediaAI/multimediaAI-1.jpg',
    technologies: 'GANs, RNNs, EfficientNet, ResNet, WaveNet, SincNet, PyTorch, TensorFlow, OpenCV, FFmpeg, Docker',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '7', // Multimedia AI
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
    {
    id: '34',
    title: 'Adaptive Industrial Voice Assistant with Real-Time Cloning & Lip Sync',
    description: 'Create an AI-powered Digital Operator for industrial environments (like manufacturing floors or remote sites). The key feature is real-time voice cloning (to give the assistant a familiar team lead\'s voice) and lip-sync animation (for visual interfaces like smart glasses or monitoring screens) to deliver complex instructions, warnings, or operational data. This improves human-machine communication, personalization, and accessibility, especially for multilingual teams.',
    image: '/AI_Portfolio/mediaAI/multimediaAI-2.png',
    technologies: 'Tacotron 2, FastSpeech 2, WaveNet, GE2E, Wav2Lip, TensorRT, Quantization, Python, SCADA',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '7', // Multimedia AI
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
     {
    id: '35',
    title: 'Generative AI for Predictive Soundscape and Anomaly Modeling',
    description: 'The system learns the normal, healthy operational soundscape. It then uses this model to synthesize new, realistic \'near-failure\' or \'anomaly\' sounds to proactively train a simpler, real-time anomaly detection system and to test human operator hearing thresholds.',
    image: '/AI_Portfolio/mediaAI/multimediaAI-3.png',
    technologies: 'VAEs, WaveGAN 2, NSynth, MFCCs, Wav2Lip, Industrial Server',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '7', // Multimedia AI
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
       {
    id: '36',
    title: 'Edge AI for Real-time Quality Control in High-Speed Manufacturing',
    description: 'Implementation a highly efficient, real-time quality inspection system directly on the manufacturing line using Edge AI.',
    image: '/AI_Portfolio/edgeAI/edgeAI-1.png',
    technologies: 'NVIDIA Jetson Orin Nano, YOLOv8-Nano, EfficientDet-Lite, MobileNetV3-SSD, Quantization, Pruning, TensorFlow Lite, PyTorch Mobile, NVIDIA TensorRT, OpenCV, Basler, Hikrobot, GigE Vision, Custom PLC, MQTT, Custom Dashboard',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '10', // Edge AI
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
       {
    id: '37',
    title: 'Edge AI for Predictive Maintenance of Remote Infrastructure (Oil & Gas, Utilities)',
    description: 'This project focuses on deploying AI models on low-power edge devices in remote, often harsh, industrial environments such as oil pipelines, wind farms, electrical substations, remote pumping stations.',
    image: '/AI_Portfolio/edgeAI/edgeAI-2.png',
    technologies: 'ESP32-S3, Raspberry Pi with Coral TPU, Autoencoders, One-Class SVMs, simple LSTMs, ensorFlow Lite Micro, Edge Impulse, OpenMV, Accelerometers, Temperature Sensors, Pressure Transducers, Acoustic Sensors, LoRaWAN, NB-IoT, cellular (LTE-M, 5G), Custom Solar Panels, Battery Packs',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '10', // Edge AI
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
    {
    id: '38',
    title: 'Edge AI for Worker Safety and Situational Awareness in Construction/Logistics',
    description: 'This project aims to enhance worker safety by deploying Edge AI-powered cameras in dynamic industrial environments like construction sites, warehouses, or loading docks, continuously monitoring predefined zones and worker locations to detect potential hazards in real-time.',
    image: '/AI_Portfolio/edgeAI/edgeAI-3.png',
    technologies: 'Object Detection, Worker Localization & Tracking, Proximity Alerts, PPE (Personal Protective Equipment) Compliance, NVIDIA Jetson Orin Nano Super, YOLO, Faster R-CNN, OpenPose, MediaPipe Pose, PyTorch, TensorFlow, NVIDIA DeepStream SDK, OpenCV, RTSP, MQTT, WiFi/Ethernet',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '10', // Edge AI
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
   {
    id: '39',
    title: 'Automated MLOps Pipeline for Predictive Maintenance',
    description: 'This project focuses on building a fully automated MLOps pipeline for predictive maintenance (PdM) on critical manufacturing equipment such as pumps, motors, CNC machines.',
    image: '/AI_Portfolio/mlOps/mlops-1.png',
    technologies: 'Apache Airflow, MLflow, Git, Docker, K8s, Triton Inference Server, Prometheus',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '8', // Machine Learning Operations (MLOps)
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  {
    id: '40',
    title: 'Feature Store for Cross-Plant Quality Control Models',
    description: 'This project aims to implement a Feature Store to support multiple machine learning models used for real-time quality inspection across different factories or, production lines.',
    image: '/AI_Portfolio/mlOps/mlops-2.png',
    technologies: 'WS SageMaker Feature Store, Apache Spark, Redis, Data Lake',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '8', // Machine Learning Operations (MLOps)
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  {
    id: '41',
    title: 'Serverless MLOps for Edge-to-Cloud Logistics Optimization',
    description: 'This project involves deploying small, high-frequency machine learning models such as models predicting delivery times, optimizing vehicle routing, or forecasting parcel volume across a vast logistics network.',
    image: '/AI_Portfolio/mlOps/mlops-3.png',
    technologies: 'Azure Functions, Azure ML, Terraform, MLflow, Azure Storage',
    // github: 'https://github.com/example/wix-website',
    // demo: 'https://business-site.wixsite.com',Multi-Modal Learning
    featured: false,
    published: true,
    categoryId: '8', // Machine Learning Operations (MLOps)
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
]

// Helper functions to get data
export const getProjects = (): Project[] => {
  return projects.filter(project => project.published)
}

export const getCategories = (): Category[] => {
  return categories
}

export const getProjectById = (id: string): Project | undefined => {
  return projects.find(project => project.id === id)
}

export const getCategoryById = (id: string): Category | undefined => {
  return categories.find(category => category.id === id)
}
