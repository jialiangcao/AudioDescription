import { createClient, type SupabaseClient } from "@supabase/supabase-js";

// Both are inlined at build time, so they must be set per Vercel environment
// rather than at runtime.
const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL ?? "";
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY ?? "";

/** True when the app is configured to sign users in. */
export const authEnabled = Boolean(SUPABASE_URL && SUPABASE_ANON_KEY);

let client: SupabaseClient | null = null;

export function supabase(): SupabaseClient {
  if (!authEnabled) {
    throw new Error(
      "Supabase is not configured: set NEXT_PUBLIC_SUPABASE_URL and " +
        "NEXT_PUBLIC_SUPABASE_ANON_KEY",
    );
  }
  if (!client) {
    client = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
  }
  return client;
}

/**
 * The current access token, or null when signed out.
 *
 * Returns null (rather than throwing) when auth is not configured at all, so a
 * local backend running with ADESC_DEV_USER_ID works without a Supabase project.
 */
export async function accessToken(): Promise<string | null> {
  if (!authEnabled) return null;
  const { data } = await supabase().auth.getSession();
  return data.session?.access_token ?? null;
}
