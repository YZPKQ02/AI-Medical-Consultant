"""Create the initial normalized consultation schema."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("users"):
        op.create_table(
            "users",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("display_name", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("deleted_at", sa.DateTime(timezone=True)),
        )
    if not inspector.has_table("consultations"):
        op.create_table(
            "consultations",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("owner_user_id", sa.Text(), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("chief_complaint", sa.Text(), nullable=False),
            sa.Column("user_context_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("deleted_at", sa.DateTime(timezone=True)),
        )
    if not inspector.has_table("messages"):
        op.create_table(
            "messages",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("consultation_id", sa.Text(), sa.ForeignKey("consultations.id", ondelete="CASCADE"), nullable=False),
            sa.Column("sequence", sa.Integer(), nullable=False),
            sa.Column("role", sa.Text(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("payload_json", postgresql.JSONB(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("deleted_at", sa.DateTime(timezone=True)),
        )
    if not inspector.has_table("agent_runs"):
        op.create_table(
            "agent_runs",
            sa.Column("id", sa.Text(), primary_key=True),
            sa.Column("consultation_id", sa.Text(), sa.ForeignKey("consultations.id", ondelete="CASCADE"), nullable=False),
            sa.Column("assistant_message_id", sa.Text(), sa.ForeignKey("messages.id", ondelete="CASCADE"), nullable=False),
            sa.Column("intent", sa.Text(), nullable=False),
            sa.Column("stage", sa.Text(), nullable=False),
            sa.Column("risk_level", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("department", sa.Text(), nullable=False),
            sa.Column("analysis_json", postgresql.JSONB(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
    if not inspector.has_table("tool_calls"):
        op.create_table(
            "tool_calls",
            sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
            sa.Column("agent_run_id", sa.Text(), sa.ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("consultation_id", sa.Text(), sa.ForeignKey("consultations.id", ondelete="CASCADE"), nullable=False),
            sa.Column("name", sa.Text(), nullable=False),
            sa.Column("status", sa.Text(), nullable=False),
            sa.Column("input_summary_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("output_summary_json", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    op.execute("CREATE INDEX IF NOT EXISTS idx_consultations_owner_active_updated ON consultations(owner_user_id, deleted_at, updated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_messages_consultation_sequence ON messages(consultation_id, sequence)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_agent_runs_consultation ON agent_runs(consultation_id, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_consultation ON tool_calls(consultation_id, created_at DESC)")


def downgrade() -> None:
    op.drop_table("tool_calls")
    op.drop_table("agent_runs")
    op.drop_table("messages")
    op.drop_table("consultations")
    op.drop_table("users")
