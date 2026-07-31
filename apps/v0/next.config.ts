import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  transpilePackages: ['@handfirst/charts', '@handfirst/datasets'],
};

export default nextConfig;
