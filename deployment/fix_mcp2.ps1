$lines = Get-Content 'c:\proj\ldt\MCP-SERVERS.md' -ReadCount 0
$out = @()
for ($i = 0; $i -lt $lines.Length; $i++) {
    $l = $lines[$i]
    $next = if ($i+1 -lt $lines.Length) { $lines[$i+1] } else { '' }
    $prev = if ($i-1 -ge 0) { $lines[$i-1] } else { '' }
    $next2 = if ($i+2 -lt $lines.Length) { $lines[$i+2] } else { '' }
    $next3 = if ($i+3 -lt $lines.Length) { $lines[$i+3] } else { '' }
    $next4 = if ($i+4 -lt $lines.Length) { $lines[$i+4] } else { '' }
    # Remove PostgreSQL row from table
    if ($l -match '^\|\s+\*\*PostgreSQL\*\*\s+\|') { continue }
    # Remove PostgreSQL workflow block (6 lines)
    if ($l -match 'User: "The pipeline seems slow\. Can you analyze the database performance\?"') { continue }
    if ($prev -match 'User: "The pipeline seems slow\. Can you analyze the database performance\?"') { continue }
    if ($l -match '^\s*→\s+Runs health checks on postgres$') { continue }
    if ($l -match '^\s*→\s+Identifies slow queries via pg_stat_statements$') { continue }
    if ($l -match '^\s*→\s+Suggests indexes using hypopg simulation$') { continue }
    if ($prev -match '^\s*→\s+Identifies slow queries via pg_stat_statements$') { continue }
    # Remove Database Performance Analysis section (header + 5 code lines)
    if ($l -match '^### Database Performance Analysis$') { continue }
    if ($l -match '^```bash$') {
        if ($next -match '^# Via postgres-mcp' -or $next -match '"Run analyze_db_health') { continue }
        if ($prev -match '^### Database Performance Analysis$') { continue }
    }
    if ($l -match '^# Via postgres-mcp') { continue }
    if ($l -match '"Run analyze_db_health') { continue }
    if ($l -match '"Analyze the top 10 slowest queries') { continue }
    if ($l -match '"Help me write an EXPLAIN plan') { continue }
    if ($l -match '^```$') {
        if ($prev -match '"Run analyze_db_health' -or $prev -match '"Analyze the top 10 slowest') { continue }
    }
    # Remove PostgreSQL troubleshooting
    if ($l -match '^#### PostgreSQL MCP: connection timeout through pgbouncer$') { continue }
    if ($l -match '^```bash$') {
        if ($next -match '# Try direct connection to postgres') { continue }
    }
    if ($l -match '# Try direct connection to postgres') { continue }
    if ($l -match '# Change DATABASE_URI to:') { continue }
    if ($l -match '^postgresql://cadqstream:cadqstream123@localhost:5432') { continue }
    if ($l -match '^$') {
        if ($prev -match 'localhost:5432/dq_pipeline$') { continue }
    }
    # Remove postgres-mcp upgrade
    if ($l -match '^# postgres-mcp$') { continue }
    if ($l -match '^uv tool upgrade postgres-mcp$') { continue }
    # Remove PostgreSQL reference link
    if ($l -match '^\- \[PostgreSQL MCP\]') { continue }
    $out += $l
}
$out | Set-Content 'c:\proj\ldt\MCP-SERVERS.md'
