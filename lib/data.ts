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
    name: 'Voice Cloning and Lip Sync',
    description: 'Voice cloning and lip sync technologies for creating realistic voice and facial animations for various applications including virtual assistants, gaming, entertainment, and more.',
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
    id: '1',
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
    id: '2',
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
    id: '3',
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
    id: '4',
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
    id: '5',
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
    id: '6',
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
    id: '7',
    title: 'Machine Learning Dashboard',
    description: 'A comprehensive dashboard for machine learning model training, monitoring, and deployment with real-time analytics. Features model versioning, performance metrics, and automated retraining pipelines.',
    image: '/placeholder.svg',
    technologies: 'Python, TensorFlow, React, D3.js, FastAPI, Docker',
    github: 'https://github.com/example/ml-dashboard',
    demo: 'https://ml-dashboard.vercel.app',
    featured: false,
    published: true,
    categoryId: '7', // AI/ML
    createdAt: '2024-02-10T00:00:00Z',
    updatedAt: '2024-02-10T00:00:00Z',
  },
  {
    id: '8',
    title: 'IoT Home Automation',
    description: 'Smart home automation system controlling lights, temperature, and security through IoT devices and mobile app. Features voice control, scheduling, and energy monitoring with real-time notifications.',
    image: '/placeholder.svg',
    technologies: 'React Native, Arduino, MQTT, Node.js, AWS IoT, Alexa API',
    github: 'https://github.com/example/iot-home',
    demo: 'https://iot-demo.vercel.app',
    featured: false,
    published: false,
    categoryId: '4', // Mobile
    createdAt: '2024-02-15T00:00:00Z',
    updatedAt: '2024-02-15T00:00:00Z',
  },
  {
    id: '9',
    title: 'Cryptocurrency Tracker',
    description: 'Real-time cryptocurrency price tracking with portfolio management, price alerts, and market analysis tools. Features include portfolio analytics, price predictions, and news integration.',
    image: '/placeholder.svg',
    technologies: 'Vue.js, Chart.js, CoinGecko API, PWA, WebSocket, IndexedDB',
    github: 'https://github.com/example/crypto-tracker',
    demo: 'https://crypto-tracker.vercel.app',
    featured: false,
    published: true,
    categoryId: '1', // Frontend
    createdAt: '2024-02-20T00:00:00Z',
    updatedAt: '2024-02-20T00:00:00Z',
  },
  {
    id: '10',
    title: 'Social Media Analytics',
    description: 'Advanced analytics platform for social media performance tracking with sentiment analysis and engagement metrics. Integrates with Twitter, Instagram, and LinkedIn APIs for comprehensive insights.',
    image: '/placeholder.svg',
    technologies: 'Next.js, Python, Twitter API, PostgreSQL, Redis, Chart.js',
    github: 'https://github.com/example/social-analytics',
    demo: 'https://social-analytics.vercel.app',
    featured: false,
    published: false,
    categoryId: '10', // Data Science
    createdAt: '2024-02-25T00:00:00Z',
    updatedAt: '2024-02-25T00:00:00Z',
  },
  {
    id: '11',
    title: 'Real Estate Platform',
    description: 'A comprehensive real estate platform with property listings, virtual tours, and mortgage calculators. Features include advanced search filters, map integration, and agent management system.',
    image: '/placeholder.svg',
    technologies: 'React, Node.js, MongoDB, Google Maps API, Stripe, Cloudinary',
    github: 'https://github.com/example/real-estate',
    demo: 'https://real-estate-demo.vercel.app',
    featured: false,
    published: true,
    categoryId: '3', // Full-Stack
    createdAt: '2024-03-01T00:00:00Z',
    updatedAt: '2024-03-01T00:00:00Z',
  },
  {
    id: '12',
    title: 'Fitness Tracking App',
    description: 'A comprehensive fitness tracking application with workout plans, nutrition tracking, and progress analytics. Features include wearable device integration, social challenges, and personalized recommendations.',
    image: '/placeholder.svg',
    technologies: 'React Native, Node.js, PostgreSQL, Firebase, HealthKit API',
    github: 'https://github.com/example/fitness-tracker',
    demo: 'https://fitness-demo.vercel.app',
    featured: false,
    published: false,
    categoryId: '4', // Mobile
    createdAt: '2024-03-05T00:00:00Z',
    updatedAt: '2024-03-05T00:00:00Z',
  },
  {
    id: '13',
    title: 'RESTful API Service',
    description: 'A comprehensive REST API service for managing user data, authentication, and business logic. Features include rate limiting, API documentation, and comprehensive error handling.',
    image: '/placeholder.svg',
    technologies: 'Node.js, Express, JWT, MongoDB, Swagger, Redis',
    github: 'https://github.com/example/rest-api',
    demo: 'https://api-docs.vercel.app',
    featured: false,
    published: true,
    categoryId: '5', // API
    createdAt: '2024-03-10T00:00:00Z',
    updatedAt: '2024-03-10T00:00:00Z',
  },
  {
    id: '14',
    title: 'WordPress E-commerce Site',
    description: 'A complete e-commerce website built with WordPress and WooCommerce. Features include custom themes, payment integration, inventory management, and SEO optimization.',
    image: '/placeholder.svg',
    technologies: 'WordPress, WooCommerce, PHP, MySQL, Stripe, Elementor',
    github: 'https://github.com/example/wordpress-store',
    demo: 'https://wordpress-store.vercel.app',
    featured: false,
    published: true,
    categoryId: '6', // No-Code
    createdAt: '2024-03-15T00:00:00Z',
    updatedAt: '2024-03-15T00:00:00Z',
  },
  {
    id: '15',
    title: 'Shopify Store Setup',
    description: 'Complete Shopify store setup with custom theme development, product management, and marketing automation. Includes custom apps and integrations.',
    image: '/placeholder.svg',
    technologies: 'Shopify, Liquid, JavaScript, Shopify API, Klaviyo, Zapier',
    github: 'https://github.com/example/shopify-store',
    demo: 'https://shopify-store.myshopify.com',
    featured: false,
    published: true,
    categoryId: '6', // No-Code
    createdAt: '2024-03-20T00:00:00Z',
    updatedAt: '2024-03-20T00:00:00Z',
  },
  {
    id: '16',
    title: 'Wix Business Website',
    description: 'Professional business website built with Wix featuring custom design, contact forms, booking system, and analytics integration.',
    image: '/placeholder.svg',
    technologies: 'Wix, Wix Code, JavaScript, Google Analytics, Mailchimp',
    github: 'https://github.com/example/wix-website',
    demo: 'https://business-site.wixsite.com',
    featured: false,
    published: true,
    categoryId: '6', // No-Code
    createdAt: '2024-03-25T00:00:00Z',
    updatedAt: '2024-03-25T00:00:00Z',
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
