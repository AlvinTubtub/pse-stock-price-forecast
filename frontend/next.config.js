/** @type {import('next').NextConfig} */
const nextConfig = {
  // No image optimization backend needed on Vercel free tier; we don't
  // use next/image with remote sources here.
  images: { unoptimized: true },
  eslint: { ignoreDuringBuilds: true },
};

module.exports = nextConfig;
