-- Jobs whose source video is fetched by a worker rather than uploaded by the
-- browser (POST /api/jobs/from-url). Additive and nullable, so the running code
-- — which never selects or writes it — keeps working through a rolling deploy.
alter table public.jobs
    add column if not exists source_url text;
