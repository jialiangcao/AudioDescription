-- Job state for the adesc backend.
--
-- Replaces the in-memory JobStore + per-job status.json that only worked in a
-- single uvicorn process. Four tables, all owned by a Supabase auth user and
-- all protected by RLS so the browser can never read another user's job.
--
-- The event log is split from the timeline on purpose: the old store rewrote
-- the entire snapshot (timeline included) on every published event, which on a
-- long video meant hundreds of full serializations. Here an event is one small
-- insert and the timeline is written only when it actually changes.

create extension if not exists "pgcrypto";

-- ---------------------------------------------------------------------------
-- jobs
-- ---------------------------------------------------------------------------

create table if not exists public.jobs (
    id           uuid primary key default gen_random_uuid(),
    owner_id     uuid not null references auth.users (id) on delete cascade,
    -- created -> queued -> processing -> done | error | interrupted
    status       text not null default 'created',
    -- Last stage entered, for progress display and for debugging a stuck job.
    stage        text,
    error        text,
    filename     text,
    -- Blob key of the uploaded source, e.g. "source.mp4" (job-relative).
    source_key   text,
    duration_sec double precision,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now(),
    -- Worker lease. A processing job whose heartbeat has gone stale was on a
    -- machine that died; the reaper requeues it. This is what replaces the old
    -- startup recover() scan, which could only mark such jobs "interrupted".
    heartbeat_at timestamptz
);

create index if not exists jobs_owner_created_idx
    on public.jobs (owner_id, created_at desc);

-- Drives both the per-user quota check and the stale-lease reaper.
create index if not exists jobs_status_heartbeat_idx
    on public.jobs (status, heartbeat_at);

-- ---------------------------------------------------------------------------
-- job_events — the append-only progress log replayed to reconnecting clients
-- ---------------------------------------------------------------------------

create table if not exists public.job_events (
    job_id     uuid not null references public.jobs (id) on delete cascade,
    -- Per-job monotonic sequence. A websocket reconnect asks for `seq > n`
    -- instead of replaying from zero, which is all the old store could do.
    seq        bigint not null,
    type       text not null,
    payload    jsonb not null,
    created_at timestamptz not null default now(),
    primary key (job_id, seq)
);

-- ---------------------------------------------------------------------------
-- timelines — one row per job, rewritten only when the timeline changes
-- ---------------------------------------------------------------------------

create table if not exists public.timelines (
    job_id     uuid primary key references public.jobs (id) on delete cascade,
    doc        jsonb not null,
    updated_at timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- qa_runs — one row per asked question
-- ---------------------------------------------------------------------------

create table if not exists public.qa_runs (
    id         uuid primary key default gen_random_uuid(),
    job_id     uuid not null references public.jobs (id) on delete cascade,
    owner_id   uuid not null references auth.users (id) on delete cascade,
    question   text not null,
    -- queued -> running -> completed | failed
    status     text not null default 'queued',
    answer     text,
    reason     text,
    cycles     integer,
    history    jsonb,
    error      text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists qa_runs_job_created_idx
    on public.qa_runs (job_id, created_at desc);

-- ---------------------------------------------------------------------------
-- Row-level security
--
-- The backend connects as the service role and bypasses these, but it always
-- scopes its own queries by owner_id. The policies are what make it safe to
-- expose the tables to the browser (PostgREST / Realtime) directly.
-- ---------------------------------------------------------------------------

alter table public.jobs       enable row level security;
alter table public.job_events enable row level security;
alter table public.timelines  enable row level security;
alter table public.qa_runs    enable row level security;

create policy jobs_owner_rw on public.jobs
    for all using (owner_id = auth.uid()) with check (owner_id = auth.uid());

create policy qa_runs_owner_rw on public.qa_runs
    for all using (owner_id = auth.uid()) with check (owner_id = auth.uid());

-- job_events and timelines have no owner column of their own; they inherit
-- ownership through the job they belong to.
create policy job_events_owner_read on public.job_events
    for select using (
        exists (
            select 1 from public.jobs j
            where j.id = job_events.job_id and j.owner_id = auth.uid()
        )
    );

create policy timelines_owner_read on public.timelines
    for select using (
        exists (
            select 1 from public.jobs j
            where j.id = timelines.job_id and j.owner_id = auth.uid()
        )
    );

-- ---------------------------------------------------------------------------
-- updated_at maintenance
-- ---------------------------------------------------------------------------

create or replace function public.touch_updated_at() returns trigger
language plpgsql as $$
begin
    new.updated_at := now();
    return new;
end;
$$;

create trigger jobs_touch_updated_at
    before update on public.jobs
    for each row execute function public.touch_updated_at();

create trigger qa_runs_touch_updated_at
    before update on public.qa_runs
    for each row execute function public.touch_updated_at();
