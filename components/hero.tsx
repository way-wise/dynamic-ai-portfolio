import { Button } from "@/components/ui/button";
import {
  ArrowRight,
  Github,
  Linkedin,
  Mail,
  Twitter,
  Download,
} from "lucide-react";
import Image from "next/image";
import { TypingAnimation } from "@/components/typing-animation";

export function Hero() {
  return (
    <section
      className="relative min-h-screen pt-32 pb-12 flex items-center justify-center px-4 sm:px-6 lg:px-8 border-b border-border overflow-hidden z-0 bg-cover bg-no-repeat"
      style={{
        backgroundImage: "url(/hero-bg.png)",
        backgroundSize: "cover",
        backgroundPosition: "center",
      }}
    >
      <div className="container mx-auto relative z-10 lg:h-[85vh] flex flex-col gap-4">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left Column - Content */}
          <div className="space-y-8 text-center lg:text-left">
            <div className="space-y-6">
              <div
                className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 border border-white/50 text-white text-sm font-medium"
                data-aos="fade-up"
                data-aos-delay="100"
              >
                <div className="w-2 h-2 bg-gray-50 rounded-full animate-pulse"></div>
                Available for new opportunities
              </div>

              <h1
                className="text-4xl text-white sm:text-5xl lg:text-6xl font-bold tracking-tight text-balance"
                data-aos="fade-up"
                data-aos-delay="200"
              >
                Hi, I'm Mr. Firoz Bari
              </h1>

              <h2
                className="text-2xl sm:text-3xl lg:text-4xl font-semibold text-[#FFCA0B]"
                data-aos="fade-up"
                data-aos-delay="300"
              >
                <TypingAnimation
                  words={[
                    "Lead AI/ML Specialist",
                    "Deep Learning Engineer",
                    "Time Series Analyst",
                    "Multi-Modal AI Specialist",
                  ]}
                  className="bg-gradient-to-r from-[#FFCA0B] to-[#f1ce4c] bg-clip-text text-transparent"
                />
              </h2>

              <p
                className="text-lg sm:text-xl text-gray-100 max-w-2xl mx-auto lg:mx-0 text-balance leading-relaxed"
                data-aos="fade-up"
                data-aos-delay="400"
              >
                Principal AI/ML Specialist and Deep Learning Engineer with 12+
                years of experience accelerating innovation across the full
                machine learning lifecycle, from Multi-Modal AI and Time Series
                forecasting to Edge AI deployment and AI-Driven Automation.
              </p>
            </div>

            <div
              className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4"
              data-aos="fade-up"
              data-aos-delay="500"
            >
              <Button size="lg" asChild className="w-full sm:w-auto">
                <a href="#projects">
                  Take a look at my work
                  <ArrowRight className="ml-2 h-4 w-4" />
                </a>
              </Button>
              <Button
                size="lg"
                variant="outline"
                asChild
                className="w-full sm:w-auto"
              >
                <a href="#contact">Reach Out</a>
              </Button>
            </div>
          </div>

          {/* Right Column - Image */}
          <div
            className="flex flex-col justify-center items-center"
            data-aos="fade-left"
            data-aos-delay="400"
          >
            <div className="relative">
              {/* <div className="absolute inset-0 bg-gradient-to-r from-purple-300 to-purple-200 rounded-full blur-3xl scale-110 animate-pulse"></div> */}
              <div className="relative w-60 h-60 sm:w-80 sm:h-80 rounded-full overflow-hidden border-4 border-purple-500 shadow-2xl">
                <Image
                  src="/firoz_bari.svg"
                  alt="Firoz Bari - Full-Stack Developer & AI/ML Specialist"
                  fill
                  className="object-cover"
                  priority
                />
              </div>
              <div
                className="absolute -bottom-4 -right-4 w-24 h-24 bg-purple-600/10 rounded-full flex items-center justify-center backdrop-blur-sm border border-purple-600/50 animate-bounce"
                style={{ animationDelay: "1s" }}
              >
                <div className="text-3xl">🧠</div>
              </div>
              <div
                className="absolute -top-4 -left-4 w-20 h-20 bg-purple-600/10 rounded-full flex items-center justify-center backdrop-blur-sm border border-purple-600/50 animate-bounce"
                style={{ animationDelay: "2s" }}
              >
                <div className="text-3xl">🤖</div>
              </div>
            </div>
          </div>
          {/* Learn More About Me Section - Highlighted Box */}
        </div>
        <div className="grow mt-auto grid grid-cols-1 lg:grid-cols-2 items-end gap-4">
          <div className="pt-8" data-aos="fade-up" data-aos-delay="700">
            <div className="relative p-4 bg-[#3BE0A6] max-w-[555px] rounded-2xl">
              <div className="relative">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-1 h-6 bg-gradient-to-b from-purple-400 to-purple-600 rounded-full"></div>
                  <h3 className="text-xl font-semibold text-[#1B3447]">
                    Learn More About Me
                  </h3>
                </div>

                <p className="text-base text-[#1B3447] mb-3 text-center lg:text-left line-clamp-2">
                  I define AI technical strategy and mentor engineers,
                  leveraging over 10 years of experience to turn complex,
                  high-impact challenges into simple, scalable, and efficient
                  user-focused AI products.
                </p>

                <div className="flex flex-wrap items-center justify-center lg:justify-start gap-3">
                  <a
                    href="/about"
                    className="inline-flex items-center gap-2 text-[#1B3447] px-3 py-1.5 text-sm font-medium  rounded-md border-none !bg-[#ABF5FF]"
                  >
                    Learn More
                    <ArrowRight className="h-4 w-4" />
                  </a>
                </div>
              </div>
            </div>
          </div>
          <div
            className="pt-8 flex lg:justify-end"
            data-aos="fade-up"
            data-aos-delay="700"
          >
            <div className="relative p-4 rounded-2xl bg-[#FFCA0B] max-w-[555px]">
              <div className="relative">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-1 h-6 bg-gradient-to-b from-purple-400 to-purple-600 rounded-full"></div>
                  <h3 className="text-xl font-semibold text-black">
                    My Experience & Skills
                  </h3>
                </div>

                <p className="text-base text-black mb-3 text-center lg:text-left line-clamp-2">
                  With over 10 years of experience building and scaling
                  end-to-end AI/ML systems, I translate complex models into
                  robust, high-performance production features that deliver
                  impactful digital solutions.
                </p>

                <div className="flex flex-wrap items-center justify-center lg:justify-start gap-3">
                  <a
                    href="/skills"
                    className="inline-flex items-center gap-2 text-black px-3 py-1.5 text-sm font-medium  rounded-md border-none !bg-[#ABF5FF]"
                  >
                    Explore My Skills <ArrowRight className="h-4 w-4" />
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
