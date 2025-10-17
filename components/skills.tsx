import { Card } from "@/components/ui/card"
import { Code2, Database, Layout, Server, Smartphone, Wrench } from "lucide-react"

const skillCategories = [

{
  title: "Deep Learning",
  icon: Layout,
  skills: [
    "🖼️ CNNs (Vision)", "📝 RNNs (Sequences)", "🧠 LSTMs (Memory)", 
    "🎯 Transformers (Attention)", "🎨 GANs (Generation)", "🔍 Autoencoders (Compression)", 
    "🏗️ DBNs (Deep Belief)", "🚀 Ultralytics (YOLO)"
  ],
},
{
  title: "Time Series Analysis and Forecasting",
  icon: Server,
  skills: [
    "📈 ARIMA (Classic)", "🔄 SARIMA (Seasonal)", "📊 ETS (Smoothing)", 
    "🔮 Prophet (Facebook)", "🧠 GRU (Gated)", "🏛️ Informer (Long Sequence)", 
    "🧩 PatchTST (Patches)", "🌳 XGBoost (Gradient Boost)", "📐 SVR (Support Vectors)"
  ],
},
{
  title: "Multi-Modal Learning",
  icon: Database,
  skills: [
    "👁️🗨️ BLIP (Vision-Language)", "🐱 Gato (Generalist)", "💎 Gemini (Google)", 
    "🦩 Flamingo (Visual Dialog)", "🎨 DALL·E (Image Gen)", "📎 CLIP (Contrastive)", 
    "🗣️ PaLI (Pathways)", "🎬 VideoGPT (Video)"
  ],
},

  // {
  //   title: "Deep Learning",
  //   icon: Layout,
  //   skills: ["CNNs", "RNNs", "LSTMs", "Transformers", "GANs", "Autoencoders", "DBNs", "Ultralytics"],
  // },
  // {
  //   title: "Time Series Analysis and Forecasting",
  //   icon: Server,
  //   skills: ["ARIMA", "SARIMA", "ETS", "Prophet", "GRU", "Informer", "PatchTST", "XGBoost", "SVR"],
  // },
  // {
  //   title: "Multi-Modal Learning",
  //   icon: Database,
  //   skills: ["BLIP", "Gato", "Gemini", "Flamingo", "DALL·E", "CLIP", "PaLI", "VideoGPT"],
  // },


  // {
  //   title: "Knowledge Distillation",
  //   icon: Code2,
  //   skills: ["DistilBERT", "TinyBERT", "Patient Knowledge Distillation", "Attention Transfer", "FitNets", "Hinton's Original KD", "Data-Free KD", "MobileNet + KD"],
  // },
  // {
  //   title: "Reinforcement Learning",
  //   icon: Smartphone,
  //   skills: ["Value-Based Methods: Q-Learning, SARSA, Deep Q-Networks (DQN)", "Policy-Based Methods: REINFORCE, Proximal Policy Optimization (PPO), Trust Region Policy Optimization (TRPO)", "Actor-Critic Methods (Hybrid): Asynchronous Advantage Actor-Critic (A3C), Deep Deterministic Policy Gradient (DDPG), Soft Actor-Critic (SAC)","Model-Based Methods: Dyna, Monte Carlo Tree Search"],
  // },
  // {
  //   title: "Federated Learning",
  //   icon: Wrench,
  //   skills: ["Federated Averaging (FedAvg)", "Federated Stochastic Gradient Descent (FedSGD)", "FedProx", "FedMA", "FedDyn", "FedOpt", "MOON", "Secure Aggregation", "Differential Privacy", "Personalized Federated Learning", "Hierarchical Federated Learning"],
  // },

  // {
  //   title: "Multimedia AI",
  //   icon: Wrench,
  //   skills: ["Vidnoz", "Latent Sync", "Wav2Lip", "Make-A-Video (Meta)", "Fish Speech V1.5", "CosyVoice2-0.5B", "IndexTTS-2", "Llasa-3B"],
  // },
  // {
  //   title: "Machine Learning Operations (MLOps)",
  //   icon: Wrench,
  //   skills: ["Azure DevOps", "Azure Pipelines", "mlflow", "Kubeflow", "TensorFlow Extended (TFX)", "Docker", "BentoML", "Kubernetes", "CI/CD Pipelines", "lakeFS", "Model Monitoring", "Data Versioning", "GCP", "AWS"],
  // },
  // {
  //   title: "Quantum Machine Learning (QML)",
  //   icon: Wrench,
  //   skills: ["Quantum Kernel Methods", "Variational Quantum Classifiers (VQC)", "Quantum GANs (QGANs)", "Quantum Boltzmann Machines", "Qiskit (IBM)", "APennyLane (Xanadu)", "TensorFlow Quantum", "Cirq (Google)", "PyQuil (Rigetti)"],
  // },
  // {
  //   title: "Edge AI",
  //   icon: Wrench,
  //   skills: ["TensorFlow Lite", "ONNX Runtime", "NVIDIA TensorRT", "OpenVINO Toolkit"],
  // },
]

export function Skills() {
  return (
    <section id="skills" className="py-24 px-4 sm:px-6 lg:px-8 bg-green-50">
      <div className="container mx-auto max-w-6xl">
        <h2 className="text-3xl sm:text-4xl font-bold mb-12 text-gray-800">Skills & Technologies</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {skillCategories.map((category, index) => {
            const Icon = category.icon
            return (
              <Card key={index} className="p-6 shadow-none hover:shadow-lg transition-shadow bg-white border-gray-100">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-2 bg-accent/10 rounded-lg">
                    <Icon className="h-5 w-5 text-accent" />
                  </div>
                  <h3 className="text-xl font-semibold text-black">{category.title}</h3>
                </div>
                <ul className="space-y-2">
                  {category.skills.map((skill) => (
                    <li key={skill} className="text-gray-700">
                      • {skill}
                    </li>
                  ))}
                </ul>
              </Card>
            )
          })}
        </div>
      </div>
    </section>
  )
}
