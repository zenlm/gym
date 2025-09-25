# Gym Documentation Site

Modern documentation site for Gym by Zoo Labs Foundation, built with:
- **Next.js 14** with App Router
- **React 19 RC** for latest features
- **Tailwind CSS 3.4** for styling
- **Framer Motion** for animations
- **TypeScript** for type safety

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Export static site
npm run export
```

## 📦 Deployment

The site is configured to deploy automatically to **gym.zoo.ngo** via:

1. **Vercel** (Primary)
   - Auto-deploys on push to main branch
   - Custom domain: gym.zoo.ngo
   - Edge functions for optimal performance

2. **GitHub Pages** (Backup)
   - Static export available
   - Can be enabled as fallback

## 🎨 Features

- **Black Monochromatic Theme**: Sleek Zoo Labs branding
- **Matrix Rain Effect**: Subtle animated background
- **Responsive Design**: Mobile-first approach
- **Fast Performance**: Optimized with Next.js
- **SEO Optimized**: Meta tags and OpenGraph
- **Accessibility**: WCAG compliant

## 📁 Structure

```
docs/
├── app/                 # Next.js App Router
│   ├── layout.tsx      # Root layout with metadata
│   ├── page.tsx        # Homepage
│   └── globals.css     # Global styles
├── components/         # React components
├── content/           # Markdown documentation
├── public/           # Static assets
└── vercel.json      # Deployment config
```

## 🔧 Configuration

### Domain Setup

1. Add CNAME record:
   - Type: CNAME
   - Name: gym
   - Value: cname.vercel-dns.com

2. Configure in Vercel:
   - Add custom domain: gym.zoo.ngo
   - SSL certificate auto-provisioned

### Environment Variables

Create `.env.local` for local development:

```env
NEXT_PUBLIC_SITE_URL=https://gym.zoo.ngo
NEXT_PUBLIC_GA_ID=G-XXXXXXXXXX
```

## 📝 License

Copyright 2025 Zoo Labs Foundation Inc.
Apache License 2.0