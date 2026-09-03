/** @type {import('next').NextConfig} */
const nextConfig = {
  // No image optimization backend needed on Vercel free tier; we don't
  // use next/image with remote sources here.
  images: { unoptimized: true },
  eslint: { ignoreDuringBuilds: true },
  async redirects() {
    return [
      {
        source: "/learn",
        destination: "/learn-stocks",
        permanent: true,
      },
    ];
  },
};

module.exports = nextConfig;
