import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  transpilePackages: [
    '@handfirst/charts',
    '@handfirst/datasets',
    '@handfirst/utils',
  ],
};

export default nextConfig;
