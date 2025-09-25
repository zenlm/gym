'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Link from 'next/link'
import { 
  ChevronRightIcon, 
  CommandLineIcon, 
  CpuChipIcon, 
  CubeIcon,
  SparklesIcon,
  RocketLaunchIcon,
  BeakerIcon,
  ChartBarIcon,
  CodeBracketIcon,
  DocumentTextIcon,
  ArrowDownTrayIcon
} from '@heroicons/react/24/outline'

export default function Home() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  const features = [
    {
      icon: CpuChipIcon,
      title: 'BitDelta Quantization',
      description: '1-bit quantization with 10-25× memory reduction',
      highlight: '25.6× compression',
    },
    {
      icon: CubeIcon,
      title: 'DeltaSoup Aggregation',
      description: 'Byzantine-robust community model improvements',
      highlight: 'Community-driven',
    },
    {
      icon: SparklesIcon,
      title: 'GRPO/GSPO Training',
      description: 'State-of-the-art algorithms with 40-60% memory savings',
      highlight: '60% less memory',
    },
    {
      icon: RocketLaunchIcon,
      title: 'Unsloth Optimization',
      description: '2-3× training speedup with optimized kernels',
      highlight: '3× faster',
    },
    {
      icon: BeakerIcon,
      title: 'Zen Model Family',
      description: 'Qwen3-based models from 0.6B to 480B parameters',
      highlight: '5 model sizes',
    },
    {
      icon: ChartBarIcon,
      title: 'Production Ready',
      description: 'Battle-tested at scale with comprehensive tooling',
      highlight: 'Enterprise grade',
    },
  ]

  const codeExample = `# Install Gym
pip install gym-ai

# Train with BitDelta quantization
from gym import Trainer, BitDeltaConfig

trainer = Trainer(
  model="Qwen/Qwen3-4B",
  algorithm="gspo",
  quantization=BitDeltaConfig(bits=1)
)

trainer.train("alpaca_gpt4_en")
trainer.export("model.gguf")  # 10× smaller!`

  if (!mounted) return null

  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Matrix rain background */}
      <div className="fixed inset-0 matrix-rain opacity-10" />
      
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center"
          >
            <h1 className="text-6xl sm:text-7xl lg:text-8xl font-bold mb-6">
              <span className="gradient-text glow">Gym</span>
            </h1>
            <p className="text-xl sm:text-2xl text-zinc-400 mb-8">
              AI Training Platform by Zoo Labs Foundation
            </p>
            <p className="text-lg text-zinc-500 mb-12 max-w-3xl mx-auto">
              Complete training platform with BitDelta quantization, DeltaSoup aggregation,
              and state-of-the-art optimization techniques. Train AI models at the speed of thought.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/docs" className="button-primary inline-flex items-center">
                Get Started
                <ChevronRightIcon className="ml-2 h-5 w-5" />
              </Link>
              <Link href="https://github.com/zooai/gym" className="button-secondary inline-flex items-center">
                View on GitHub
                <CodeBracketIcon className="ml-2 h-5 w-5" />
              </Link>
            </div>
          </motion.div>

          {/* Terminal Demo */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="mt-20"
          >
            <div className="terminal max-w-4xl mx-auto">
              <div className="flex items-center gap-2 mb-4 pb-4 border-b border-green-500/20">
                <div className="w-3 h-3 rounded-full bg-red-500" />
                <div className="w-3 h-3 rounded-full bg-yellow-500" />
                <div className="w-3 h-3 rounded-full bg-green-500" />
                <span className="ml-4 text-xs text-zinc-500">gym-cli</span>
              </div>
              <pre className="text-sm overflow-x-auto">
                <code>{`$ gym train models/nano/configs/gspo_training.yaml

[INFO] Loading Qwen/Qwen3-0.6B model...
[INFO] Applying BitDelta quantization...
[INFO] Memory reduction: 25.6×
[INFO] Training with GSPO algorithm...
[INFO] Speed: 2.3× faster with Unsloth
[SUCCESS] Model saved to saves/zen-nano-gspo/`}</code>
              </pre>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.h2
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="text-4xl font-bold text-center mb-16 gradient-text"
          >
            Production-Ready Features
          </motion.h2>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 * index }}
                className="card group hover:border-green-500/30"
              >
                <feature.icon className="h-10 w-10 text-green-500 mb-4" />
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-zinc-400 mb-4">{feature.description}</p>
                <div className="inline-flex items-center text-green-400 text-sm font-mono">
                  <SparklesIcon className="h-4 w-4 mr-1" />
                  {feature.highlight}
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Code Example */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <h2 className="text-4xl font-bold text-center mb-16 gradient-text">
              Simple yet Powerful
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div>
                <h3 className="text-2xl font-semibold mb-4">Quick Start</h3>
                <p className="text-zinc-400 mb-6">
                  Get started with Gym in minutes. Our intuitive API makes training
                  state-of-the-art models as simple as a few lines of code.
                </p>
                <ul className="space-y-3 text-zinc-300">
                  <li className="flex items-start">
                    <ChevronRightIcon className="h-5 w-5 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                    <span>Automatic quantization with BitDelta</span>
                  </li>
                  <li className="flex items-start">
                    <ChevronRightIcon className="h-5 w-5 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                    <span>Community model aggregation with DeltaSoup</span>
                  </li>
                  <li className="flex items-start">
                    <ChevronRightIcon className="h-5 w-5 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                    <span>Export to GGUF, ONNX, or TorchScript</span>
                  </li>
                  <li className="flex items-start">
                    <ChevronRightIcon className="h-5 w-5 text-green-500 mt-0.5 mr-2 flex-shrink-0" />
                    <span>Built-in safety and jailbreak prevention</span>
                  </li>
                </ul>
              </div>

              <div className="code-block">
                <pre className="text-green-400">
                  <code>{codeExample}</code>
                </pre>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.5 }}
            className="card bg-gradient-to-br from-zinc-900 to-black border-green-500/20"
          >
            <h2 className="text-3xl font-bold mb-4">
              Ready to Train at the Speed of Thought?
            </h2>
            <p className="text-zinc-400 mb-8">
              Join the community building the future of AI training.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/docs/quickstart" className="button-primary inline-flex items-center">
                <DocumentTextIcon className="mr-2 h-5 w-5" />
                Read the Docs
              </Link>
              <Link href="/download" className="button-secondary inline-flex items-center">
                <ArrowDownTrayIcon className="mr-2 h-5 w-5" />
                Download Gym
              </Link>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative py-12 px-4 sm:px-6 lg:px-8 border-t border-zinc-800">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-zinc-500">
            © 2025 Zoo Labs Foundation Inc. All rights reserved.
          </p>
          <div className="mt-4 flex justify-center gap-6">
            <Link href="https://github.com/zooai/gym" className="text-zinc-400 hover:text-green-400 transition-colors">
              GitHub
            </Link>
            <Link href="https://zoo.ngo" className="text-zinc-400 hover:text-green-400 transition-colors">
              Zoo Labs
            </Link>
            <Link href="/docs" className="text-zinc-400 hover:text-green-400 transition-colors">
              Documentation
            </Link>
          </div>
        </div>
      </footer>
    </main>
  )
}