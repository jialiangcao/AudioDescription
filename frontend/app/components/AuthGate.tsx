"use client";

import { createContext, useContext, useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { authEnabled, supabase } from "../lib/supabase";

/** Who is signed in, for the header's account menu. Null when auth is off. */
const AccountContext = createContext<{ email: string | null }>({ email: null });

export function useAccount() {
  return useContext(AccountContext);
}

/** Ends the session. Safe to call when auth isn't configured — it does nothing. */
export async function signOut() {
  if (!authEnabled) return;
  await supabase().auth.signOut();
}

/**
 * Gates the app behind a Supabase session.
 *
 * When Supabase isn't configured this renders its children unchanged, so a
 * backend running with ADESC_DEV_USER_ID works without a Supabase project.
 * Sign-in is a magic link: no passwords to store, and it matches the fact that
 * a job's only real identity requirement is a stable owner.
 */
export default function AuthGate({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(!authEnabled);
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!authEnabled) return;
    const client = supabase();
    client.auth.getSession().then(({ data }) => {
      setSession(data.session);
      setReady(true);
    });
    const { data: subscription } = client.auth.onAuthStateChange(
      (_event, next) => setSession(next),
    );
    return () => subscription.subscription.unsubscribe();
  }, []);

  if (!authEnabled) {
    return (
      <AccountContext.Provider value={{ email: null }}>
        {children}
      </AccountContext.Provider>
    );
  }

  if (!ready) {
    return (
      <div className="gate">
        <p className="gate-loading">Getting things ready…</p>
      </div>
    );
  }

  if (!session) {
    return (
      <div className="gate">
        <div className="gate-card">
          <p className="wordmark gate-mark">BuddyWatch</p>
          <h1 className="gate-title">Audio description for any video</h1>
          <p className="gate-copy">
            Sign in and we&rsquo;ll email you a link — no password to remember.
          </p>
          {sent ? (
            <p className="gate-sent">
              Link sent. Check <strong>{email}</strong> and open it on this
              device.
            </p>
          ) : (
            <form
              className="gate-form"
              onSubmit={async (e) => {
                e.preventDefault();
                setError(null);
                // Send the link back to whichever deployment the user is on.
                // Without this, Supabase uses the project's single Site URL, so
                // signing in from a preview deploy would land on production.
                const { error: signInError } =
                  await supabase().auth.signInWithOtp({
                    email,
                    options: { emailRedirectTo: window.location.origin },
                  });
                if (signInError) setError(signInError.message);
                else setSent(true);
              }}
            >
              <input
                type="email"
                required
                placeholder="you@example.com"
                aria-label="Email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
              <button type="submit" className="btn-primary">
                Email me a link
              </button>
            </form>
          )}
          {error && <p className="notice notice-error">{error}</p>}
        </div>
      </div>
    );
  }

  return (
    <AccountContext.Provider value={{ email: session.user.email ?? null }}>
      {children}
    </AccountContext.Provider>
  );
}
