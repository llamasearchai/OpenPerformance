"""Initial schema creation

Revision ID: 001
Revises: 
Create Date: 2025-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('role', sa.Enum('ADMIN', 'USER', 'VIEWER', 'SERVICE', name='userrole'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('totp_secret', sa.String(length=32), nullable=True),
        sa.Column('api_key', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('password_changed_at', sa.DateTime(), nullable=True),
        sa.Column('failed_login_attempts', sa.Integer(), nullable=False),
        sa.Column('locked_until', sa.DateTime(), nullable=True),
        sa.Column('preferences', sa.Text(), nullable=True),
        sa.Column('metadata', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_user_created_at', 'users', ['created_at'], unique=False)
    op.create_index('idx_user_email_active', 'users', ['email', 'is_active'], unique=False)
    op.create_index('idx_user_role_active', 'users', ['role', 'is_active'], unique=False)
    op.create_index(op.f('ix_users_api_key'), 'users', ['api_key'], unique=True)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # Create organizations table
    op.create_table('organizations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('display_name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('website', sa.String(length=500), nullable=True),
        sa.Column('settings', sa.JSON(), nullable=True),
        sa.Column('features', sa.JSON(), nullable=True),
        sa.Column('max_projects', sa.Integer(), nullable=True),
        sa.Column('max_members', sa.Integer(), nullable=True),
        sa.Column('max_storage_gb', sa.Integer(), nullable=True),
        sa.Column('billing_email', sa.String(length=255), nullable=True),
        sa.Column('subscription_tier', sa.String(length=50), nullable=True),
        sa.Column('subscription_expires_at', sa.DateTime(), nullable=True),
        sa.Column('logo_url', sa.String(length=500), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_verified', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_organization_active', 'organizations', ['is_active'], unique=False)
    op.create_index('idx_organization_tier', 'organizations', ['subscription_tier'], unique=False)
    op.create_index(op.f('ix_organizations_id'), 'organizations', ['id'], unique=False)
    op.create_index(op.f('ix_organizations_name'), 'organizations', ['name'], unique=True)

    # Create api_keys table
    op.create_table('api_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('scopes', sa.Text(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('rate_limit', sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_api_key_active', 'api_keys', ['is_active'], unique=False)
    op.create_index('idx_api_key_expires_at', 'api_keys', ['expires_at'], unique=False)
    op.create_index('idx_api_key_user_id', 'api_keys', ['user_id'], unique=False)
    op.create_index(op.f('ix_api_keys_id'), 'api_keys', ['id'], unique=False)
    op.create_index(op.f('ix_api_keys_key'), 'api_keys', ['key'], unique=True)

    # Create audit_logs table
    op.create_table('audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('resource_type', sa.String(length=100), nullable=True),
        sa.Column('resource_id', sa.String(length=100), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('details', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_audit_log_resource', 'audit_logs', ['resource_type', 'resource_id'], unique=False)
    op.create_index('idx_audit_log_timestamp', 'audit_logs', ['timestamp'], unique=False)
    op.create_index('idx_audit_log_user_action', 'audit_logs', ['user_id', 'action'], unique=False)
    op.create_index(op.f('ix_audit_logs_action'), 'audit_logs', ['action'], unique=False)
    op.create_index(op.f('ix_audit_logs_id'), 'audit_logs', ['id'], unique=False)
    op.create_index(op.f('ix_audit_logs_timestamp'), 'audit_logs', ['timestamp'], unique=False)
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)

    # Create benchmarks table
    op.create_table('benchmarks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('type', sa.Enum('TRAINING', 'INFERENCE', 'DISTRIBUTED', 'MEMORY', 'THROUGHPUT', 'LATENCY', 'CUSTOM', name='benchmarktype'), nullable=False),
        sa.Column('framework', sa.String(length=100), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('config', sa.JSON(), nullable=False),
        sa.Column('default_params', sa.JSON(), nullable=False),
        sa.Column('validation_rules', sa.JSON(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('difficulty', sa.String(length=50), nullable=True),
        sa.Column('estimated_runtime', sa.Integer(), nullable=True),
        sa.Column('min_gpu_memory_gb', sa.Float(), nullable=True),
        sa.Column('min_cpu_cores', sa.Integer(), nullable=True),
        sa.Column('min_memory_gb', sa.Float(), nullable=True),
        sa.Column('requires_cuda', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_benchmark_category', 'benchmarks', ['category'], unique=False)
    op.create_index('idx_benchmark_type_framework', 'benchmarks', ['type', 'framework'], unique=False)
    op.create_index(op.f('ix_benchmarks_framework'), 'benchmarks', ['framework'], unique=False)
    op.create_index(op.f('ix_benchmarks_id'), 'benchmarks', ['id'], unique=False)
    op.create_index(op.f('ix_benchmarks_name'), 'benchmarks', ['name'], unique=True)
    op.create_index(op.f('ix_benchmarks_type'), 'benchmarks', ['type'], unique=False)

    # Create hardware_profiles table
    op.create_table('hardware_profiles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('organization_id', sa.Integer(), nullable=True),
        sa.Column('hostname', sa.String(length=255), nullable=True),
        sa.Column('platform', sa.String(length=100), nullable=True),
        sa.Column('platform_version', sa.String(length=100), nullable=True),
        sa.Column('architecture', sa.String(length=50), nullable=True),
        sa.Column('cpu_model', sa.String(length=255), nullable=True),
        sa.Column('cpu_cores', sa.Integer(), nullable=True),
        sa.Column('cpu_threads', sa.Integer(), nullable=True),
        sa.Column('cpu_frequency_ghz', sa.Float(), nullable=True),
        sa.Column('cpu_cache_mb', sa.Float(), nullable=True),
        sa.Column('memory_total_gb', sa.Float(), nullable=True),
        sa.Column('memory_speed_mhz', sa.Integer(), nullable=True),
        sa.Column('memory_type', sa.String(length=50), nullable=True),
        sa.Column('gpu_count', sa.Integer(), nullable=True),
        sa.Column('gpu_models', sa.JSON(), nullable=True),
        sa.Column('gpu_memory_total_gb', sa.Float(), nullable=True),
        sa.Column('gpu_driver_version', sa.String(length=50), nullable=True),
        sa.Column('cuda_version', sa.String(length=50), nullable=True),
        sa.Column('storage_type', sa.String(length=50), nullable=True),
        sa.Column('storage_total_gb', sa.Float(), nullable=True),
        sa.Column('storage_speed_mbps', sa.Float(), nullable=True),
        sa.Column('network_type', sa.String(length=100), nullable=True),
        sa.Column('network_speed_gbps', sa.Float(), nullable=True),
        sa.Column('is_cloud', sa.Boolean(), nullable=True),
        sa.Column('cloud_provider', sa.String(length=100), nullable=True),
        sa.Column('instance_type', sa.String(length=100), nullable=True),
        sa.Column('cluster_name', sa.String(length=255), nullable=True),
        sa.Column('node_count', sa.Integer(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('validated', sa.Boolean(), nullable=True),
        sa.Column('validation_date', sa.DateTime(), nullable=True),
        sa.Column('benchmark_scores', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('last_seen_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name', 'user_id', name='uq_hardware_profile_name_user')
    )
    op.create_index('idx_hardware_profile_org', 'hardware_profiles', ['organization_id'], unique=False)
    op.create_index('idx_hardware_profile_platform', 'hardware_profiles', ['platform'], unique=False)
    op.create_index('idx_hardware_profile_user', 'hardware_profiles', ['user_id'], unique=False)
    op.create_index(op.f('ix_hardware_profiles_id'), 'hardware_profiles', ['id'], unique=False)
    op.create_index(op.f('ix_hardware_profiles_name'), 'hardware_profiles', ['name'], unique=False)
    op.create_index(op.f('ix_hardware_profiles_organization_id'), 'hardware_profiles', ['organization_id'], unique=False)
    op.create_index(op.f('ix_hardware_profiles_user_id'), 'hardware_profiles', ['user_id'], unique=False)

    # Create optimization_profiles table
    op.create_table('optimization_profiles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('framework', sa.String(length=100), nullable=False),
        sa.Column('model_type', sa.String(length=100), nullable=True),
        sa.Column('workload_type', sa.String(length=100), nullable=True),
        sa.Column('strategies', sa.JSON(), nullable=False),
        sa.Column('enabled_optimizations', sa.JSON(), nullable=False),
        sa.Column('memory_optimization', sa.JSON(), nullable=True),
        sa.Column('compute_optimization', sa.JSON(), nullable=True),
        sa.Column('communication_optimization', sa.JSON(), nullable=True),
        sa.Column('io_optimization', sa.JSON(), nullable=True),
        sa.Column('min_gpu_memory_gb', sa.Float(), nullable=True),
        sa.Column('recommended_gpu_models', sa.JSON(), nullable=True),
        sa.Column('target_throughput', sa.Float(), nullable=True),
        sa.Column('target_latency_ms', sa.Float(), nullable=True),
        sa.Column('target_memory_reduction', sa.Float(), nullable=True),
        sa.Column('validated', sa.Boolean(), nullable=True),
        sa.Column('validation_results', sa.JSON(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_optimization_profile_framework', 'optimization_profiles', ['framework'], unique=False)
    op.create_index('idx_optimization_profile_model', 'optimization_profiles', ['model_type'], unique=False)
    op.create_index(op.f('ix_optimization_profiles_framework'), 'optimization_profiles', ['framework'], unique=False)
    op.create_index(op.f('ix_optimization_profiles_id'), 'optimization_profiles', ['id'], unique=False)
    op.create_index(op.f('ix_optimization_profiles_name'), 'optimization_profiles', ['name'], unique=True)

    # Create projects table
    op.create_table('projects',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('slug', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=True),
        sa.Column('visibility', sa.Enum('PRIVATE', 'INTERNAL', 'PUBLIC', name='projectvisibility'), nullable=False),
        sa.Column('is_template', sa.Boolean(), nullable=True),
        sa.Column('template_id', sa.Integer(), nullable=True),
        sa.Column('settings', sa.JSON(), nullable=True),
        sa.Column('default_benchmark_params', sa.JSON(), nullable=True),
        sa.Column('max_concurrent_runs', sa.Integer(), nullable=True),
        sa.Column('max_storage_gb', sa.Integer(), nullable=True),
        sa.Column('total_runs', sa.Integer(), nullable=True),
        sa.Column('successful_runs', sa.Integer(), nullable=True),
        sa.Column('failed_runs', sa.Integer(), nullable=True),
        sa.Column('total_runtime_hours', sa.Float(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('readme', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('is_archived', sa.Boolean(), nullable=False),
        sa.Column('archived_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('last_activity_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], ),
        sa.ForeignKeyConstraint(['template_id'], ['projects.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('slug', 'organization_id', name='uq_project_slug_org'),
        sa.UniqueConstraint('slug', 'user_id', name='uq_project_slug_user')
    )
    op.create_index('idx_project_active', 'projects', ['is_active'], unique=False)
    op.create_index('idx_project_org', 'projects', ['organization_id'], unique=False)
    op.create_index('idx_project_user', 'projects', ['user_id'], unique=False)
    op.create_index('idx_project_visibility', 'projects', ['visibility'], unique=False)
    op.create_index(op.f('ix_projects_id'), 'projects', ['id'], unique=False)
    op.create_index(op.f('ix_projects_name'), 'projects', ['name'], unique=False)
    op.create_index(op.f('ix_projects_organization_id'), 'projects', ['organization_id'], unique=False)
    op.create_index(op.f('ix_projects_slug'), 'projects', ['slug'], unique=False)
    op.create_index(op.f('ix_projects_user_id'), 'projects', ['user_id'], unique=False)

    # Create refresh_tokens table
    op.create_table('refresh_tokens',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('token', sa.String(length=500), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('revoked', sa.Boolean(), nullable=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_refresh_token_expires_at', 'refresh_tokens', ['expires_at'], unique=False)
    op.create_index('idx_refresh_token_revoked', 'refresh_tokens', ['revoked'], unique=False)
    op.create_index('idx_refresh_token_user_id', 'refresh_tokens', ['user_id'], unique=False)
    op.create_index(op.f('ix_refresh_tokens_id'), 'refresh_tokens', ['id'], unique=False)
    op.create_index(op.f('ix_refresh_tokens_token'), 'refresh_tokens', ['token'], unique=True)
    op.create_index(op.f('ix_refresh_tokens_user_id'), 'refresh_tokens', ['user_id'], unique=False)

    # Create remaining tables with foreign keys
    # ... (continuing with the rest of the tables)


def downgrade() -> None:
    # Drop all tables in reverse order
    op.drop_table('refresh_tokens')
    op.drop_table('projects')
    op.drop_table('optimization_profiles')
    op.drop_table('hardware_profiles')
    op.drop_table('benchmarks')
    op.drop_table('audit_logs')
    op.drop_table('api_keys')
    op.drop_table('organizations')
    op.drop_table('users')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS userrole')
    op.execute('DROP TYPE IF EXISTS benchmarktype')
    op.execute('DROP TYPE IF EXISTS projectvisibility')