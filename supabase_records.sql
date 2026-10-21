create table if not exists records (
  id bigint generated always as identity primary key,
  title text not null,
  details text not null
);

alter table records enable row level security;

drop policy if exists "anon can read records" on records;

create policy "anon can read records"
  on records
  for select
  to anon
  using (true);

insert into records (title, details)
values (
  'Remote days',
  'Employees may work remotely up to three days per week.'
);
