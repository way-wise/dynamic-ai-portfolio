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
  clientLocation?: string
  clientType?: string // e.g., "Military", "Enterprise", "Third Party Vendor", "Government", "Startup"
  projectDuration?: string // e.g., "3 months", "6 months", "1 year"
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
]

// Projects data
export const projects: Project[] = [
  // Deep Learning Projects
  {
    id: '46',
    title: 'Damaged Freight Label Processor',
    description: 'Never lose a package to a bad label again. This specialized system is designed to rescue data from compromised shipping documents, whether they are wrinkled, smudged, or ripped. By accurately isolating and reading damaged text and barcodes, it provides uninterrupted data flow for inventory management and package routing, even under poor conditions.',
    image: '/AI_Portfolio/dl/dl-2.png',
    technologies: 'Python, OpenCV, PyTorch, Customly Trained Hybrid Transformer, Scikit-image, PaddleOCR, Custom Data, Offline Custom Server for Deployment, FastAPI, Object Detection, Image Classification',
    featured: false,
    published: true,
    categoryId: '1', // Deep Learning
    clientLocation: 'Sydney, Australia',
    clientType: 'Enterprise',
    projectDuration: '5 months',
    createdAt: '2024-01-10T00:00:00Z',
    updatedAt: '2024-01-10T00:00:00Z',
  },
  {
    id: '66',
    title: 'Document Classification and Data Extraction System',
    description: 'Built an intelligent document processing system that automatically classifies document types and extracts key information from invoices, receipts, and contracts using deep learning models.',
    image: '/AI_Portfolio/mediaFresh/6.png',
    technologies: 'Python, PyTorch, Transformers, OpenCV, Tesseract OCR, FastAPI, PostgreSQL',
    featured: false,
    published: true,
    categoryId: '1', // Deep Learning
    clientLocation: 'London, UK',
    clientType: 'Enterprise',
    projectDuration: '4 months',
    createdAt: '2024-04-01T00:00:00Z',
    updatedAt: '2024-04-01T00:00:00Z',
  },
  {
    id: '67',
    title: 'Fashion Item Recognition and Recommendation',
    description: 'Developed a computer vision system that identifies fashion items from images and provides personalized recommendations based on user preferences and style patterns.',
    image: '/AI_Portfolio/mediaFresh/7.png',
    technologies: 'Python, TensorFlow, ResNet, EfficientNet, Flask, Redis, AWS S3',
    featured: false,
    published: true,
    categoryId: '1', // Deep Learning
    clientLocation: 'New York, USA',
    clientType: 'Startup',
    projectDuration: '3 months',
    createdAt: '2024-04-05T00:00:00Z',
    updatedAt: '2024-04-05T00:00:00Z',
  },
  {
    id: '68',
    title: 'Medical Image Analysis for Disease Detection',
    description: 'Created a deep learning model to assist radiologists in detecting abnormalities in medical images, improving diagnostic accuracy and reducing analysis time.',
    image: '/AI_Portfolio/mediaFresh/8.png',
    technologies: 'Python, PyTorch, U-Net, DenseNet, Medical Imaging Libraries, Docker, HIPAA Compliance',
    featured: true,
    published: true,
    categoryId: '1', // Deep Learning
    clientLocation: 'Boston, USA',
    clientType: 'Healthcare Provider',
    projectDuration: '6 months',
    createdAt: '2024-04-10T00:00:00Z',
    updatedAt: '2024-04-10T00:00:00Z',
  },
  {
    id: '74',
    title: 'Real-time Object Detection for Security Systems',
    description: 'Developed a high-performance object detection system for security cameras that identifies people, vehicles, and suspicious activities in real-time with 95% accuracy.',
    image: '/AI_Portfolio/mediaFresh/9.png',
    technologies: 'Python, YOLOv8, OpenCV, TensorRT, NVIDIA Jetson, Flask, WebRTC',
    featured: false,
    published: true,
    categoryId: '1', // Deep Learning
    clientLocation: 'Dubai, UAE',
    clientType: 'Security Company',
    projectDuration: '5 months',
    createdAt: '2024-05-10T00:00:00Z',
    updatedAt: '2024-05-10T00:00:00Z',
  },
  {
    id: '75',
    title: 'Automated Quality Control for Manufacturing',
    description: 'Built a computer vision system that automatically detects defects in manufactured products, reducing quality control time by 70% and improving detection accuracy.',
    image: '/AI_Portfolio/mediaFresh/10.png',
    technologies: 'Python, PyTorch, CNN, OpenCV, FastAPI, Docker, Industrial Cameras',
    featured: true,
    published: true,
    categoryId: '1', // Deep Learning
    clientLocation: 'Seoul, South Korea',
    clientType: 'Manufacturing Enterprise',
    projectDuration: '4 months',
    createdAt: '2024-05-15T00:00:00Z',
    updatedAt: '2024-05-15T00:00:00Z',
  },
  {
    id: '76',
    title: 'Facial Recognition Attendance System',
    description: 'Created an automated attendance system using facial recognition that eliminates manual tracking and provides real-time attendance analytics for organizations.',
    image: '/AI_Portfolio/mediaFresh/11.png',
    technologies: 'Python, FaceNet, MTCNN, Django, React, PostgreSQL, AWS EC2',
    featured: false,
    published: true,
    categoryId: '1', // Deep Learning
    clientLocation: 'Singapore',
    clientType: 'Education Technology',
    projectDuration: '3 months',
    createdAt: '2024-05-20T00:00:00Z',
    updatedAt: '2024-05-20T00:00:00Z',
  },

  // Time Series Projects
  {
    id: '47',
    title: 'Automated Inventory Demand Predictor',
    description: 'Predicts exactly how much product customers will buy at different times and locations.',
    image: '/AI_Portfolio/time-series/time-series-4.jpeg',
    technologies: 'FastAPI, Custom Dashboard with Streamlit, Custom service integrated with email, AWS Lambda, AWS Step Functions, Docker, MLflow, Exponential Smoothing, Auto-ARIMA, Prophet from Meta, CatBoost, TimescaleDB, Warehouse Management Systems',
    featured: false,
    published: true,
    categoryId: '2', // Time Series
    clientLocation: 'Singapore',
    clientType: 'Enterprise',
    projectDuration: '7 months',
    createdAt: '2024-01-05T00:00:00Z',
    updatedAt: '2024-01-05T00:00:00Z',
  },
  {
    id: '48',
    title: 'Predictive Maintenance Platform',
    description: 'Uses time series sensor data from critical machines to forecast RUL.',
    image: '/AI_Portfolio/time-series/time-series-3.jpeg',
    technologies: 'Python 3.10, Pandas, Scikit-learn, MLflow, LSTMs, GRUs, TensorFlow, PyTorch, Gradient Boosting, Grafana, Django, React, Amazon S3',
    featured: false,
    published: true,
    categoryId: '2', // Time Series
    clientLocation: 'Tokyo, Japan',
    clientType: 'Enterprise',
    projectDuration: '9 months',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '50',
    title: 'Forecasting Application for Fisheries Production',
    description: 'Prediction of fisheries production to provide insights on fisheries economy.',
    image: '/AI_Portfolio/time-series/time-series-1.jpeg',
    technologies: 'Python 3.9, Prophet, SARIMAX, XGBoost, FastAPI, Docker',
    featured: false,
    published: true,
    categoryId: '2', // Time Series
    clientLocation: 'Oslo, Norway',
    clientType: 'Government',
    projectDuration: '6 months',
    createdAt: '2024-02-05T00:00:00Z',
    updatedAt: '2024-02-05T00:00:00Z',
  },
  {
    id: '69',
    title: 'Energy Consumption Forecasting System',
    description: 'Developed a time series forecasting system to predict energy consumption patterns for utility companies, enabling better resource allocation and cost optimization.',
    image: '/AI_Portfolio/mediaFresh/5.png',
    technologies: 'Python, Prophet, LSTM, XGBoost, Plotly Dash, InfluxDB, AWS',
    featured: false,
    published: true,
    categoryId: '2', // Time Series
    clientLocation: 'Berlin, Germany',
    clientType: 'Enterprise',
    projectDuration: '5 months',
    createdAt: '2024-04-15T00:00:00Z',
    updatedAt: '2024-04-15T00:00:00Z',
  },
  {
    id: '70',
    title: 'E-commerce Sales Prediction Dashboard',
    description: 'Built a comprehensive dashboard that predicts future sales trends, seasonal patterns, and customer demand for an e-commerce platform.',
    image: '/AI_Portfolio/mediaFresh/4.png',
    technologies: 'Python, ARIMA, LightGBM, Streamlit, PostgreSQL, Docker',
    featured: false,
    published: true,
    categoryId: '2', // Time Series
    clientLocation: 'San Francisco, USA',
    clientType: 'Startup',
    projectDuration: '4 months',
    createdAt: '2024-04-20T00:00:00Z',
    updatedAt: '2024-04-20T00:00:00Z',
  },

  // Multi-Modal Learning Projects
  {
    id: '23',
    title: 'Multilingual LLM Chatbot for Customer Support',
    description: 'Created a real-time multilingual LLM chatbot capable of handling support queries in 15+ languages with sentiment-aware response tuning.',
    image: '/AI_Portfolio/llm/l5.png',
    technologies: 'Python 3.8, PyTorch, MarianMT, Redis, FastAPI, Kubernetes',
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    clientLocation: 'Dublin, Ireland',
    clientType: 'Enterprise',
    projectDuration: '5 months',
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  {
    id: '24',
    title: 'RAG-Based Enterprise Knowledge Assistant',
    description: 'Built a context-aware RAG assistant to answer internal employee queries by grounding LLM responses in enterprise documents, policies, and knowledge bases.',
    image: '/AI_Portfolio/llm/l4.png',
    technologies: 'Python 3.9, PyTorch, Hugging Face Transformers, LangChain, FAISS, Pinecone, FastAPI, Docker',
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    clientLocation: 'Seattle, USA',
    clientType: 'Enterprise',
    projectDuration: '9 months',
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  {
    id: '25',
    title: 'LLM-based Content Generation Engine',
    description: 'Engineered a scalable content generation pipeline for SEO blogs, product descriptions, and brand copy using GPT-J and T5 models.',
    image: '/AI_Portfolio/llm/l6.png',
    technologies: 'Python 3.8, Hugging Face Transformers, GPT-J, T5, Gradio, Docker',
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    clientLocation: 'Austin, USA',
    clientType: 'Startup',
    projectDuration: '4 months',
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  {
    id: '26',
    title: 'SmartLegalBot - AI Legal Assistant for Contract Review',
    description: 'SmartLegalBot is an AI-powered assistant that helps users review legal contracts, highlight risky clauses, suggest simplified rewrites, and answer basic legal queries, all using a fine-tuned LLM trained on legal documents.',
    image: '/AI_Portfolio/ai/llm.png',
    technologies: 'Node.js, Hugging Face Transformers, PostgreSQL',
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    clientLocation: 'Chicago, USA',
    clientType: 'Enterprise',
    projectDuration: '7 months',
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  {
    id: '29',
    title: 'Autonomous Research Agent using LLMs',
    description: 'Developed an autonomous LLM agent for literature reviews with semantic paper search, PDF summarization, and research graph generation.',
    image: '/AI_Portfolio/llm/l7.png',
    technologies: 'Python 3.10, LangChain, Transformers, PyMuPDF, Neo4j',
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    clientLocation: 'Copenhagen, Denmark',
    clientType: 'Enterprise',
    projectDuration: '6 months',
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  {
    id: '30',
    title: 'LLM-Powered Code Generation & Review Assistant',
    description: 'Built an AI coding assistant that generates boilerplate code, reviews PRs, and flags anti-patterns with LLM-powered insights.',
    image: '/AI_Portfolio/llm/l9.png',
    technologies: 'Python 3.10, CodeLLaMA, GitHub API, FastAPI, Docker',
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    clientLocation: 'Tel Aviv, Israel',
    clientType: 'Startup',
    projectDuration: '5 months',
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
  },
  {
    id: '71',
    title: 'Image Captioning and Visual Question Answering System',
    description: 'Developed a multi-modal AI system that generates descriptive captions for images and answers questions about visual content, enhancing accessibility and content understanding.',
    image: '/AI_Portfolio/mediaFresh/1.png',
    technologies: 'Python, PyTorch, CLIP, BLIP, Transformers, FastAPI, React',
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    clientLocation: 'Toronto, Canada',
    clientType: 'Media Company',
    projectDuration: '4 months',
    createdAt: '2024-04-25T00:00:00Z',
    updatedAt: '2024-04-25T00:00:00Z',
  },
  {
    id: '72',
    title: 'Multi-Modal Product Search Engine',
    description: 'Created a search engine that allows users to find products using both text queries and image uploads, improving e-commerce search experience.',
    image: '/AI_Portfolio/mediaFresh/2.png',
    technologies: 'Python, Sentence Transformers, CLIP, Elasticsearch, FastAPI, AWS',
    featured: true,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    clientLocation: 'Amsterdam, Netherlands',
    clientType: 'E-commerce Platform',
    projectDuration: '5 months',
    createdAt: '2024-05-01T00:00:00Z',
    updatedAt: '2024-05-01T00:00:00Z',
  },
  {
    id: '73',
    title: 'Audio-Visual Content Moderator',
    description: 'Built a content moderation system that analyzes both audio and visual content to detect inappropriate material across social media platforms.',
    image: '/AI_Portfolio/mediaFresh/3.png',
    technologies: 'Python, PyTorch, Whisper, CLIP, OpenCV, FastAPI, Redis',
    featured: false,
    published: true,
    categoryId: '3', // Multi-Modal Learning
    clientLocation: 'Austin, USA',
    clientType: 'Social Media Startup',
    projectDuration: '6 months',
    createdAt: '2024-05-05T00:00:00Z',
    updatedAt: '2024-05-05T00:00:00Z',
  }
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