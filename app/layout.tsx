import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
})

const jetbrainsMono = JetBrains_Mono({ 
  subsets: ['latin'],
  variable: '--font-mono',
})

export const metadata: Metadata = {
  title: 'Gym - AI Training Platform by Zoo Labs',
  description: 'Complete AI training platform with BitDelta quantization, DeltaSoup aggregation, and state-of-the-art optimization techniques',
  keywords: 'AI, machine learning, training, quantization, BitDelta, DeltaSoup, Zoo Labs, Gym',
  authors: [{ name: 'Zoo Labs Foundation Inc.' }],
  openGraph: {
    title: 'Gym - AI Training Platform',
    description: 'Train AI models at the speed of thought',
    url: 'https://gym.zoo.ngo',
    siteName: 'Gym',
    images: [
      {
        url: 'https://gym.zoo.ngo/og.png',
        width: 1200,
        height: 630,
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Gym - AI Training Platform',
    description: 'Train AI models at the speed of thought',
    creator: '@zoolabs',
    images: ['https://gym.zoo.ngo/og.png'],
  },
  viewport: 'width=device-width, initial-scale=1',
  robots: 'index, follow',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-sans bg-black text-white antialiased`}>
        <div className="min-h-screen bg-gradient-to-b from-black via-zinc-900 to-black">
          {children}
        </div>
      </body>
    </html>
  )
}