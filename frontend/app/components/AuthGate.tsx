"use client";

import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { authEnabled, supabase } from "../lib/supabase";

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

  if (!authEnabled) return <>{children}</>;
  if (!ready) return <main className="container">Loading…</main>;

  if (!session) {
    return (
      <main className="container">
        <h1>Audio Description</h1>
        <p className="subtitle">Sign in to generate audio description.</p>
        {sent ? (
          <p>Check your email for a sign-in link.</p>
        ) : (
          <form
            className="ask-row"
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
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <button type="submit">Send sign-in link</button>
          </form>
        )}
        {error && <div className="error">{error}</div>}
      </main>
    );
  }

  return (
    <>
      <div className="auth-bar">
        <span>{session.user.email}</span>
        <button onClick={() => supabase().auth.signOut()}>Sign out</button>
      </div>
      {children}
    </>
  );
}
