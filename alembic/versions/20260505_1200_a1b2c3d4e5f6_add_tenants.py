"""add tenants
Revision ID: a1b2c3d4e5f6
Revises: 49ca957f5d06
Create Date: 2026-05-05 12:00:00.000000
"""
from __future__ import annotations
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '49ca957f5d06'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # 1. Create tenants table
    op.create_table('tenants',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('tenant_key', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('company_name', sa.String(length=512), nullable=False),
        sa.Column('company_info', sa.Text(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_tenants_tenant_key'), 'tenants', ['tenant_key'], unique=True)
    op.create_index(op.f('ix_tenants_user_id'), 'tenants', ['user_id'], unique=False)

    # 2. Create tenant_api_specs table
    op.create_table('tenant_api_specs',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('tenant_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('method', sa.String(length=32), nullable=False),
        sa.Column('url', sa.String(length=1024), nullable=False),
        sa.Column('headers', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('query_params', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('request_body', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('path_params', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('lookup_fields', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_tenant_api_specs_tenant_id'), 'tenant_api_specs', ['tenant_id'], unique=False)

    # 3. Add tenant_id to documents
    op.add_column('documents', sa.Column('tenant_id', sa.UUID(), nullable=True))
    op.create_index(op.f('ix_documents_tenant_id'), 'documents', ['tenant_id'], unique=False)
    op.create_foreign_key('fk_documents_tenant_id', 'documents', 'tenants', ['tenant_id'], ['id'], ondelete='CASCADE')

def downgrade() -> None:
    # 1. Remove from documents
    op.drop_constraint('fk_documents_tenant_id', 'documents', type_='foreignkey')
    op.drop_index(op.f('ix_documents_tenant_id'), table_name='documents')
    op.drop_column('documents', 'tenant_id')

    # 2. Drop tenant_api_specs
    op.drop_index(op.f('ix_tenant_api_specs_tenant_id'), table_name='tenant_api_specs')
    op.drop_table('tenant_api_specs')

    # 3. Drop tenants
    op.drop_index(op.f('ix_tenants_user_id'), table_name='tenants')
    op.drop_index(op.f('ix_tenants_tenant_key'), table_name='tenants')
    op.drop_table('tenants')
