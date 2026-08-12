-- Minimal stand-in for the parts of Supabase's auth schema the migrations
-- reference, so the real migrations can run against a plain Postgres (the
-- docker-compose service and CI's service container).
--
-- Only what 0001_jobs.sql needs: the auth.users table it foreign-keys to, and
-- an auth.uid() for the RLS policies to compile against. Policies are never
-- exercised here — the backend connects as the owner and bypasses RLS, exactly
-- as it does against Supabase with the service role.

create schema if not exists auth;

create table if not exists auth.users (
    id uuid primary key
);

create or replace function auth.uid() returns uuid
language sql stable as $$
    select nullif(current_setting('request.jwt.claim.sub', true), '')::uuid
$$;
