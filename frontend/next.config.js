const path = require('path');

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The app lives in a subdirectory of a mostly-Python repo; pin file tracing to
  // frontend/ so Next never infers a workspace root further up and walks it.
  outputFileTracingRoot: path.join(__dirname),
};

module.exports = nextConfig;
