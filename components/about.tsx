import { Card } from "@/components/ui/card"
import { Code2, Heart, Lightbulb, Target } from "lucide-react"

export function About() {
  return (
    <section id="about" className="py-24 px-4 sm:px-6 lg:px-8 bg-[#f7f7f7]">
      <div className="container mx-auto max-w-6xl">
        <div className="text-center mb-16">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4 text-gray-800">About Me</h2>
          <p className="text-lg text-gray-700 max-w-2xl mx-auto">
            Passionate about creating AI-driven solutions that solve complex, real-world problems.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center mb-16">
          <div className="space-y-6 text-lg leading-relaxed text-gray-700">
            <p>
              I am a Deep Learning Engineer passionate about translating theoretical models into high-impact, deployed solutions. My favorite work lies at the intersection of cutting-edge research such as Multi-Modal AI and Reinforcement Learning and robust MLOps, creating systems that are not only algorithmically powerful but are also meticulously engineered for scalability and performance.
            </p>
            <p>
              Currently, I am a Principal AI/ML Specialist specializing in architecting and deploying end-to-end machine learning pipelines. I contribute to the design and maintenance of scalable AI applications, ensuring our models adhere to best practices in AI Ethics and are optimized for both cloud and Edge AI environments to deliver powerful, inclusive insights.
            </p>
            <p>
              In the past, I've had the opportunity to develop complex predictive and automation software across a variety of settings from large Fortune 500 corporations and financial institutions to innovative deep-tech start-ups. Additionally, I have been an advocate for knowledge sharing, having previously mentored junior engineers and contributed to the field through publications on Time Series Analysis and Knowledge Distillation.
            </p>
            <p>
              In my spare time, I am usually reading research papers, exploring new algorithms like those in Quantum Machine Learning, or working on side projects that advance my skills in Federated Learning and AI-Driven Automation.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <Card className="p-6 text-center bg-green-200 border-green-300 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 bg-green-500 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Code2 className="h-6 w-6 text-white" />
              </div>
              <h3 className="font-semibold mb-2 text-black">Clean Code</h3>
              <p className="text-sm text-gray-700">
                Writing maintainable, scalable code that follows best practices
              </p>
            </Card>

            <Card className="p-6 text-center bg-blue-200 border-blue-300 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Heart className="h-6 w-6 text-white" />
              </div>
              <h3 className="font-semibold mb-2 text-black">User-Focused</h3>
              <p className="text-sm text-gray-700">
                Building experiences that users love and find intuitive
              </p>
            </Card>

            <Card className="p-6 text-center bg-yellow-200 border-yellow-300 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 bg-yellow-500 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Lightbulb className="h-6 w-6 text-white" />
              </div>
              <h3 className="font-semibold mb-2 text-black">Innovation</h3>
              <p className="text-sm text-gray-700">
                Always exploring new technologies and creative solutions
              </p>
            </Card>

            <Card className="p-6 text-center bg-red-200 border-red-300 hover:shadow-lg transition-shadow">
              <div className="w-12 h-12 bg-red-500 rounded-lg flex items-center justify-center mx-auto mb-4">
                <Target className="h-6 w-6 text-white" />
              </div>
              <h3 className="font-semibold mb-2 text-black">Results-Driven</h3>
              <p className="text-sm text-gray-700">
                Focused on delivering value and achieving project goals
              </p>
            </Card>
          </div>
        </div>
      </div>
    </section>
  )
}
